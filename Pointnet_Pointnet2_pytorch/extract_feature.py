import torch
import numpy as np
import open3d as o3d
from sklearn.manifold import TSNE

import pandas as pd
import plotly.express as px

import sys, os
import argparse
import msgspec

PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

from models.pointnet_feature import PointNetFeatureExtractor
from models.pointnet2_utils import sample_and_group


def read_and_process_ply(ply_name, num_kp, num_samples, radius, is_mesh=True, down_sample=False):
    '''
    Input 
        ply_name : Scene name of the point cloud
        num_kp : number of keypoints (the centroid of the each group)
        num_samples : number of points for each group
        radius : radius for all groups. equals to the range of sampling
        is_mesh : type of the input file (mesh or point cloud)
        down_sample : down sampling the original ply file (memory constraints)
    
    Returns
        cat_features : features including xyz, rgb, normals for each point, 
                        which will be fed into the feature extractor
        centroid : centroid coordinates&features for each group (size : [B, 9])
        fps_idx : indices for all centroid
        grouped_xyz : coordinates of points for each group
        grouped_idx : indices for points of each group
    '''
    ply_path = os.path.join(PATH, f"{ply_name}.ply")
    
    if is_mesh:
        mesh = o3d.io.read_triangle_mesh(ply_path)
        if not down_sample:
            xyz = torch.from_numpy(np.asarray(mesh.vertices).reshape(1,-1,3)).float()
            rgb = torch.from_numpy(np.asarray(mesh.vertex_colors).reshape(1,-1,3)).float()
            normal = torch.from_numpy(np.asarray(mesh.vertex_normals).reshape(1,-1,3)).float()
        else:
            pcd = mesh.sample_points_poisson_disk(10000, use_triangle_normal=True)
            xyz = torch.from_numpy(np.asarray(pcd.points).reshape(1,-1,3)).float()
            rgb = torch.from_numpy(np.asarray(pcd.colors).reshape(1,-1,3)).float()
            normal = torch.from_numpy(np.asarray(pcd.normals).reshape(1,-1,3)).float()
    else:
        pcd = o3d.io.read_point_cloud(ply_path)
        xyz = torch.from_numpy(np.asarray(pcd.points).reshape(1,-1,3)).float()
        rgb = torch.from_numpy(np.asarray(pcd.colors).reshape(1,-1,3)).float()
        normal = torch.from_numpy(np.asarray(pcd.normals).reshape(1,-1,3)).float()
    
    print(f"xyz size : {xyz.size()}")
    print(f"rgb size : {rgb.size()}")
    print(f"normal size : {normal.size()}")
    
    features = torch.cat((xyz, rgb, normal), axis=-1)
    print(f"features size : {features.size()}")

    new_xyz, new_points, grouped_xyz, fps_idx, grouped_idx = sample_and_group(num_kp, radius, 
                                                                              num_samples, xyz, features, 
                                                                              returnfps=True)
    
    centroid = features.reshape(-1,9)[fps_idx.reshape(-1)].reshape(1,num_kp,1,9)
    grouped = new_points[:,:,:,3:]
    cat_features = torch.cat((centroid, grouped), axis=2).reshape(num_kp,-1,9)

    return cat_features, centroid.squeeze(), fps_idx, grouped_xyz, grouped_idx


def extract_features(model, features, device, batch_size=16):
    model = model.to(device)
    features = features.to(device)

    with torch.no_grad():
        model.eval()
        input_feat = features.transpose(2,1)
        epochs = input_feat.size(0) // batch_size

        all_embeddings = []
        for epoch in range(epochs):
            print(f"Running epoch No. {epoch}")
            start_idx = epoch * batch_size
            end_idx = start_idx + batch_size

            embedding = model(input_feat[start_idx:end_idx])
            all_embeddings.append(embedding)

    all_embeddings = torch.cat(all_embeddings, axis=0)

    return all_embeddings

def get_label(indices, is_centroid=True, file_name="2t7WUuJeko7_points_labels.npy"):
    cat_pth = os.path.join(PATH, file_name)

    object_per_point = np.load(cat_pth)

    obj_labels = torch.zeros_like(indices)

    if is_centroid:
        batch_size, num_centriods = indices.size()

        for B in range(batch_size):
            for N in range(num_centriods):
                id = indices[B,N]
                obj_labels[B,N] = object_per_point[id.item()]

    else:
        batch_size, num_centriods, num_samples = indices.size()

        for B in range(batch_size):
            for N in range(num_centriods):
                for S in range(num_samples):
                    id = indices[B,N,S]
                    obj_labels[B,N,S] = object_per_point[id.item()]
    
    return obj_labels


def count_labels(a, return_count=False, idx=0, cat_file_name="mpcat40.tsv"):
    cat_pth = os.path.join(PATH, cat_file_name)
    labels_tsv = pd.read_csv(cat_pth, sep="\t", encoding = "utf-8")

    unique, counts = np.unique(a, return_counts=True)

    if return_count:
        return len(unique)
    else:
        uniq_cnt_dict = dict(zip(unique, counts))
        sorted_dict = sorted(uniq_cnt_dict.items(), reverse=True, key=lambda item : item[1])

        if len(unique) -1 < idx:
            return " "
        elif len(unique) + idx < 0:
            return " "
        else:
            top_object = labels_tsv.loc[sorted_dict[idx][0], 'mpcat40']
            top_object = f"{top_object:>20}"
            return top_object

def visualize(embeddings, kp_coord, kp_labels_int, samples_coord, samples_labels, cat_file_name="category_mapping.tsv", tsne_components=2):
    embeddings = embeddings.cpu().numpy()
    kp_coord = kp_coord.cpu().numpy()
    kp_labels_int = kp_labels_int.cpu().numpy()
    samples_coord = samples_coord.cpu().numpy()
    samples_labels = samples_labels.cpu().numpy()

    cat_pth = os.path.join(PATH, cat_file_name)
    labels_tsv = pd.read_csv(cat_pth, sep="\t", encoding = "utf-8")
    kp_labels = np.apply_along_axis(lambda x : labels_tsv.loc[x-1,'mpcat40'], axis=1, arr=kp_labels_int).reshape(-1)
    kp_xyz = np.apply_along_axis(lambda x : f"({x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f})", axis=1, arr=kp_coord).reshape(-1)
    top1_label = np.apply_along_axis(count_labels, arr=samples_labels, axis=-1, idx=0).reshape(-1)
    top2_label = np.apply_along_axis(count_labels, arr=samples_labels, axis=-1, idx=1).reshape(-1)
    bottom1_label = np.apply_along_axis(count_labels, arr=samples_labels, axis=-1, idx=-1).reshape(-1)
    bottom2_label = np.apply_along_axis(count_labels, arr=samples_labels, axis=-1, idx=-2).reshape(-1)
    num_nearby = np.apply_along_axis(count_labels, arr=samples_labels, axis=-1, return_count=True).reshape(-1)
    

    model = TSNE(n_components=tsne_components, perplexity=40, n_iter=5000)
    embeddings_transformed = model.fit_transform(embeddings[:,:,0])

    df = pd.DataFrame({'x' : embeddings_transformed[:,0], 'y' : embeddings_transformed[:,1], 
                       'labels' : kp_labels, 'xyz' : kp_xyz, 'num_nearby' : num_nearby,
                       'top1 nearby' : top1_label, 'top2 nearby' : top2_label, 
                       'least1 nearby' : bottom1_label, 'least2 nearby' : bottom2_label,
                       'size' : 5 * np.ones_like(embeddings_transformed.shape[0], dtype=int)})

    fig = px.scatter(df, x='x', y='y', color='labels', size='size', 
                     custom_data=['xyz', 'num_nearby', 'top1 nearby',
                                  'top2 nearby', 'least1 nearby', 'least2 nearby', 'labels'])
    
    fig.update_traces(
        hovertemplate =
                    "<b>%{customdata[6]}</b><br>" +
                    "<b>%{customdata[0]}</b><br><br>" +
                    "Number of neighbors: %{customdata[1]}<br>" +
                    "Top1 frequent neighbor: %{customdata[2]}<br>" +
                    "Top2 frequent neighbor: %{customdata[3]}<br>" +
                    "Least1 frequent neighbor: %{customdata[4]}<br>" +
                    "Least2 frequent neighbor: %{customdata[5]}<br>" +
                    "<extra></extra>"
    )

    fig.update_layout(hoverlabel_namelength=-1)

    fig.show()

def main(args):
    ply_name = args.scene_name
    num_kp = args.num_keypoints
    num_samples = args.num_samples
    radius = args.radius
    batch_size = args.batch_size
    down_sample = args.down_sample

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    PointNetEncoder = PointNetFeatureExtractor(num_kp, radius, num_samples, device)

    # 1. Read mesh file and process to get the xyz coordinates & features
    features, kp_coords, kp_idx, samples_xyz, samples_idx = read_and_process_ply(ply_name, 
                                                                                 num_kp, 
                                                                                 num_samples, 
                                                                                 radius,
                                                                                 is_mesh=True,
                                                                                 down_sample=down_sample)
    print(f"features size : {features.size()}")
    print(f"kp_coords size : {kp_coords.size()}")
    print(f"kp_idx size : {kp_idx.size()}")
    print(f"samples_xyz size : {samples_xyz.size()}")
    print(f"samples_idx size : {samples_idx.size()}")
    
    # 2. Get the object labels for each points
    kp_labels = get_label(kp_idx, is_centroid=True)
    samples_labels = get_label(samples_idx, is_centroid=False)

    print(f"kp_labels size : {kp_labels.size()}")
    print(f"samples_labels size : {samples_labels.size()}")

    # 3. Get embedding vectors of the keypoints
    all_embeddings = extract_features(PointNetEncoder, features, device, batch_size)
    print(f"embeddings size : {all_embeddings.size()}")

    # 4. Visualize the embedding
    visualize(all_embeddings, kp_coords, kp_labels, samples_xyz, samples_labels)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, default="2t7WUuJeko7")
    parser.add_argument("--num_keypoints", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=4096)
    parser.add_argument("--radius", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--down_sample", action='store_true')

    args = parser.parse_args()

    main(args)
