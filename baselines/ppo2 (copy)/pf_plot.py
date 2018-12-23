import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib import cm
from pygmo import hypervolume
import pygmo as pg


def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

# weights = []
# for _ in range(30):
#     random_w = np.random.rand(5)
#     weights.append(random_w)
# np.save('/home/xi/workspace/weights_half.npy', weights)


areas, indexs, base = [], [], []

# all_results_mp, all_results_meta = [], []
mp_full, meta_full = [], []
mp_full_valid, meta_full_valid, all_valid = [], [], []
pf_mp, pf_meta = [], []

base_mp, base_meta = 0,0

env_name = 'reacher'

valid_mp, valid_meta = 0, 0
higher_v_mp, higher_v_meta = 0, 0

for i in range(30):
    meta_result = np.load('/home/xi/workspace/exp_models/exp_'+env_name+'/meta_finetune/results_'+str(i)+'.npy')[:,-1]
    mp_result = np.load('/home/xi/workspace/exp_models/exp_'+env_name+'/mp_train/results_'+str(i)+'.npy')[:,-1]

    meta = np.load('/home/xi/workspace/exp_models/exp_'+env_name+'/meta_finetune/results_'+str(i)+'.npy')[:,0:-1]
    mp = np.load('/home/xi/workspace/exp_models/exp_'+env_name+'/mp_train/results_'+str(i)+'.npy')[:,0:-1]

    if i == 0:
        base_mp = len(mp_result)
        base_meta = len(meta_result)
    
    mp_result = mp_result[0:base_mp]
    meta_result = meta_result[0:base_meta]

    print(meta_result[-2], mp_result[-2]) 



    for j in range(-10, 0, 1):
        if mp_result[j] > meta_result[j]:
            higher_v_mp += 1
        else:
            higher_v_meta += 1

    mp_full.append(mp[0:base_mp])
    meta_full.append(meta[0:base_meta])

    start = len(mp) - 2
    for j in range(start, len(mp)-0, 1):
        sub_mp = np.append(mp[j], 1)
        mp_full_valid.append(sub_mp)
        # if sub_mp[3] > 10:
        # if sub_mp[0] > 80:
        if sub_mp[2] > -0.1:
            # mp_full_valid.append(sub_mp)
            valid_mp += 1

    start = len(meta) - 2
    for j in range(start, len(meta)-0, 1):
        sub_meta = np.append(meta[j], 0)
        meta_full_valid.append(sub_meta)
        # if sub_meta[3] > 10:
        # if sub_meta[0] > 80:
        if sub_meta[2] > -0.1:
            # meta_full_valid.append(sub_meta)
            valid_meta += 1


meta_full = np.asarray(meta_full)
mp_full = np.asarray(mp_full)
mp_full_valid = np.asarray(mp_full_valid)
meta_full_valid = np.asarray(meta_full_valid)
print(mp_full.shape)
#-----------------------------------------------------------------
size = 3
fig = plt.figure()
fig_3d_all = plt.figure()
fig_3d_sep = plt.figure()

ax_all = fig_3d_all.gca(projection='3d')
ax_sep = fig_3d_sep.gca(projection='3d')
hv_data = meta_full

hv_iteration, index, base = [], [], []
print(len(hv_data[0]), hv_data.shape)
shape = hv_data.shape
reshape_meta_full = hv_data.reshape((shape[0]*shape[1], shape[2]))
print(reshape_meta_full.shape)
min_meta = np.max(-reshape_meta_full, axis=0)
min_mp = np.max(-mp_full_valid, axis=0)
min_v = np.maximum(min_meta, min_mp[0:-1])
min_v = min_v
ref_point = min_v[0:size]
#-----------------------------------------------------------------

# min_meta = np.max(-meta_full_valid, axis=0)
# min_mp = np.max(-mp_full_valid, axis=0)
# min_v = np.maximum(min_meta, min_mp)
# min_v = min_v
# print(min_meta, min_mp, min_v)

# ref = [0, 20, 500]
hv = hypervolume(np.asarray(-meta_full_valid)[:,0:size])
ref_point = min_v[0:size]
hv_meta = hv.compute(ref_point)

hv = hypervolume(np.asarray(-mp_full_valid)[:,0:size])
ref_point = min_v[0:size]
hv_mp = hv.compute(ref_point)

# print('meta hv', np.log(hv_meta), np.log(hv_mp))
# print('meta hv', (hv_meta), (hv_mp))

# #-----------------------------------------------------------------
# for i in range(1, len(hv_data[0]), 1):
#     points = -hv_data[:, i][:,0:size]
#     hv = hypervolume(points)
#     hvi = hv.compute(ref_point)
#     hv_iteration.append(hvi)
#     index.append(i)
#     # base.append(hv_mp-30)
    
# hv_iteration = np.asarray(hv_iteration)   
# mean_hv = runningMeanFast(hv_iteration, 50)
# start_p = 10
# # plt.scatter(index[0:start_p], hv_iteration[0:start_p], c='b') 
# plt.scatter(index[0:-50], mean_hv[0:-50], c='b') 
# # plt.plot(np.asarray(index)[0:500], np.asarray(base)[0:500], 'r--', c='r')

# print('meta hv', np.max(mean_hv), (hv_mp))


# plt.title('LunarLanderContinuous')
# plt.xlabel('Fine-tuning iteration')
# plt.ylabel('Hypervolume indicator')
# #-----------------------------------------------------------------


all_valid = np.concatenate((mp_full_valid, meta_full_valid), axis=0)
min_v = np.min(all_valid, axis=0)
print(min_v)

points_mp = -np.asarray(mp_full_valid[:,0:size])
points_meta = -np.asarray(meta_full_valid[:,0:size])
points_all = -np.asarray(all_valid[:,0:size])

ndf_mp, dl, dc, ndr = pg.fast_non_dominated_sorting(points = points_mp)
ndf_meta, dl, dc, ndr = pg.fast_non_dominated_sorting(points = points_meta)
ndf_all, dl, dc, ndr = pg.fast_non_dominated_sorting(points = points_all)

f_mp = mp_full_valid[ndf_mp[0]]
f_meta = meta_full_valid[ndf_meta[0]]
f_all = all_valid[ndf_all[0]]

# plt.scatter(f_mp[:,0], f_mp[:,1], c='r')
# plt.scatter(f_meta[:,0], f_meta[:,1], c='b')
# plt.scatter(f_all[:,0], f_all[:,1], c='pink')


scale = 300
scale_start = 20
dists = np.sqrt(f_all[:, 0]*f_all[:, 0] + f_all[:, 1]*f_all[:, 1], f_all[:, 2]*f_all[:, 2])
min_dist = np.min(dists)
max_dist = np.max(dists)

mp_count, meta_count = 0, 0
dist_mp = np.sqrt(f_mp[:, 0]*f_mp[:, 0] + f_mp[:, 1]*f_mp[:, 1], f_mp[:, 2]*f_mp[:, 2])
dist_meta = np.sqrt(f_meta[:, 0]*f_meta[:, 0] + f_meta[:, 1]*f_meta[:, 1], f_meta[:, 2]*f_meta[:, 2])

ax_sep.scatter(f_mp[:,0], f_mp[:,1], f_mp[:,2], s = scale_start+ scale*(dist_mp-min_dist)/max_dist, c='r')
ax_sep.scatter(f_meta[:,0], f_meta[:,1], f_meta[:,2], s = scale_start+ scale*(dist_meta-min_dist)/max_dist, c='b')

# print(dist)
for p, d in zip(f_all, dists):
    if p[-1] == 0:
        meta_count += 1
        ax_sep.scatter(p[0], p[1], p[2], s = scale_start+ scale*((d-min_dist)/max_dist+1), facecolors='none', edgecolors='black')
        # # plt.scatter(p[0], p[1], c='b')
    else:
        mp_count += 1
        ax_sep.scatter(p[0], p[1], p[2], s = scale_start+ scale*((d-min_dist)/max_dist+1), facecolors='none', edgecolors='black')
        # # plt.scatter(p[0], p[1], c='r')

# ax_sep.scatter(min_v[0], min_v[1], min_v[2], s = 1, c='grey')
# ax_all.scatter(min_v[0], min_v[1], min_v[2], s = 1, c='grey')

ax_sep.set_title('Reacher')
ax_sep.set_xlabel('Target Distance')
ax_sep.set_ylabel('Energy Cost')
ax_sep.set_zlabel('Stuck Joint')
ax_sep.view_init(20, 45)

# ax_all.set_xlabel('Alive')
# ax_all.set_ylabel('Speed')
# ax_all.set_zlabel('Force')
# ax_all.view_init(20, 60)

# # ax.scatter(f_all[:,0], f_all[:,1], f_all[:,2], c='pink')
# # print(ndf[0])
# #-----------------------------------------------------------------
print('valid', valid_meta, valid_mp)
print('non-dominate', meta_count, mp_count)
print('higher v', higher_v_meta, higher_v_mp)
plt.show()
# # # plot_bar(pf_meta, pf_mp)
# # # plot_mesh(pf_meta, pf_mp)

# # # plot_mesh(pf_meta, pf_mp)


# mp_count, meta_count = 0, 0
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# for p in pf_all:
#     if p[-1] == 0:
#         meta_count += 1
#         ax.scatter(p[0], p[1], p[2], cmap=plt.cm.Spectral, c='b')
#         # plt.scatter(p[0], p[1], c='b')
#     else:
#         mp_count += 1
#         ax.scatter(p[0], p[1], p[2], cmap=plt.cm.Spectral, c='r')
#         # plt.scatter(p[0], p[1], c='r')
# print('valid', valid_meta, valid_mp)
# print('non-dominate', meta_count, mp_count)
# print('higher v', higher_v_meta, higher_v_mp)
# plt.show()


# show_seperate_results(all_results_meta, all_results_mp)
# show_sum_results(all_results_meta, all_results_mp)

# for i in range(1, 1000, 1):
#     pf_sets = np.load('/home/xi/workspace/model/checkpoints/pf_' + str(i) + '.npy').tolist()
#     # pf_sets = np.load('/home/xi/workspace/exp_models/exp_model_landing/meta_finetune_0/pf_' + str(i) + '.npy').tolist()
#     # pf_sets = np.load('/home/xi/workspace/exp_models/exp_model_landing/mp_training/pf_' + str(i) + '.npy').tolist()
#     # print(i, len(pf_sets))
#     if len(pf_sets) < 6:
#         continue
#     area = update_pf_sets(pf_sets, [])
#     areas.append(area)
#     indexs.append(i+1)
#     base.append(10334)

# # # # for i in range(30, 1000, 2):
# # # #     pf_sets = np.load('/home/xi/workspace/model/exp_meta_morl/pf_' + str(i) + '.npy').tolist()
# # # #     area = update_pf_sets(pf_sets, [])
# # # #     areas.append(area)
# # # #     indexs.append(i+1)
# # # #     base.append(868)

# plt.scatter(np.asarray(indexs), np.asarray(areas), c='b')
# plt.plot(np.asarray(indexs), np.asarray(base), 'r--', c='r')

# # pf_sets = np.load('/home/xi/workspace/exp_models/exp_model_landing/mp_training/pf_4300.npy').tolist()
# # plt.scatter(np.asarray(pf_sets)[:,1], np.asarray(pf_sets)[:,2], c=(1, 0.2, 0.2))

# # plt.scatter(40, -150, c='white')
# # plt.scatter(400, -20, c='white')




# area = update_pf_sets(pf_sets, [])
# print(area)