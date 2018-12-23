import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib import cm
from pygmo import hypervolume
import pygmo as pg


# weights = []
# for _ in range(30):
#     random_w = np.random.rand(5)
#     weights.append(random_w)
# np.save('/home/xi/workspace/weights_half.npy', weights)

def show_seperate_results(all_results_meta, all_results_mp):
    # all_results_mp = all_results_mp[:, :length]
    random_ws = np.load('/home/xi/workspace/weights_half.npy')
    random_ws = np.append(random_ws, [1,1,1,1,1])
    for i in range(31):
        print(i, len(all_results_mp[i]), random_ws[i])
        plt.clf()

        sum_meta = all_results_meta[i] #np.sum(all_results_meta, axis=0)
        sum_mp = all_results_mp[i] #np.sum(all_results_mp)

        # sum_mp_plot = np.full(sum_meta.shape, sum_mp)

        # print(sum_mp_plot)
        # print(sum_meta.shape, sum_mp.shape, sum_mp_plot.shape)

        plt.scatter(range(sum_meta.shape[0]), sum_meta, c=(1, 0.2, 0.2))
        plt.scatter(range(sum_mp.shape[0]), sum_mp, c=(0.2, 1, 0.2))

        plt.show()

def show_sum_results(all_results_meta, all_results_mp):
    print('sum')
    plt.clf()

    sum_meta = np.mean(all_results_meta, axis=0)
    sum_mp = np.mean(all_results_mp, axis=0)

    # sum_mp_plot = np.full(sum_meta.shape, sum_mp)

    # print(sum_mp_plot)
    # print(sum_meta.shape, sum_mp.shape, sum_mp_plot.shape)

    plt.scatter(range(sum_meta.shape[0]), sum_meta, c=(1, 0.2, 0.2))
    plt.scatter(range(sum_mp.shape[0]), sum_mp, c=(0.2, 1, 0.2))

    plt.show()


def plot_bar(meta, mp):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = meta[:,0]
    y = meta[:,1]
    z = meta[:,2]

    ax.bar(x, z, zs=y, zdir='y', color='r', alpha=0.8)

    x = mp[:,0]
    y = mp[:,1]
    z = mp[:,2]

    ax.bar(x, z, zs=y, zdir='y', color='b', alpha=0.8)    
    plt.show()

def plot_mesh(meta, mp):
    print('ploting mesh')
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    print('meta shape', meta.shape, mp.shape, np.min(mp, axis=0))
    # Make data.
    x = meta[:,0]
    y = meta[:,1]
    z = meta[:,2]

    # grid_x, grid_y = np.mgrid[min(x):max(x):200j, min(y):max(y):200j]
    # grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    # ax.plot_surface(grid_x, grid_y, grid_z, cmap=plt.cm.Spectral,
    #                     linewidth=0, antialiased=False)
    ax.scatter(x, y, z, cmap=plt.cm.Spectral, c='b')

    # ax.plot_trisurf(x, y, z, linewidth=0.01, antialiased=True)

    x = mp[:,0]
    y = mp[:,1]
    z = mp[:,2]

    # grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    # grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    # ax.plot_surface(grid_x, grid_y, grid_z, cmap=cm.coolwarm,
    #                    linewidth=0, antialiased=False)
    ax.scatter(x, y, z, cmap=plt.cm.Spectral, c='r')
    # ax.plot_trisurf(x, y, z, linewidth=0.01, antialiased=True, color='r')

    plt.show()


def update_pf_sets(pf_sets, candidate_set):
    # if candidate_set[0] < 80:
    #     return pf_sets, []

    select = [0, 1, 2]
    pf_sets_new = []
    if pf_sets == []:
        pf_sets.append(candidate_set)
    else:
        add_candidate = True
        for i in range(len(pf_sets)):
            if np.all(np.asarray(candidate_set[select]) < np.asarray(pf_sets)[i][select]): # pf_sets[i] is pareto dominated by the candidate, remove pf_sets[i] and add candidate to the pf_set
                add_candidate = False
                break
        if add_candidate:
            for i in range(len(pf_sets)):
                if np.all(np.asarray(candidate_set[select]) > np.asarray(pf_sets)[i][select]): # pf_sets[i] is pareto dominated by the candidate, remove pf_sets[i] and add candidate to the pf_set
                    pf_sets[i] = []

        if add_candidate:
            pf_sets.append(candidate_set)

        for i in range(len(pf_sets)):
            if pf_sets[i] != []:
                pf_sets_new.append(pf_sets[i])
        pf_sets = pf_sets_new*1

    # # pf_sets[-1] = np.min(pf_sets, axis=0)
    # area = 0
    pf_sets_convex = []
    # if len(pf_sets)>6:
    #     set_cpy = pf_sets*1
    #     set_cpy.append(np.min(pf_sets, axis=0))
    #     # index = ConvexHull(np.asarray(set_cpy)[:,0:3]).vertices        

    #     hull = ConvexHull(np.asarray(set_cpy)[:,select])
    #     index = hull.vertices        
    #     area = hull.area

    #     for i in index:
    #         if i < len(pf_sets):
    #             pf_sets_convex.append(pf_sets[i])

    #     # print(index, len(pf_sets)-1, add_candidate)
    #     # pf_sets = pf_sets_new


    return pf_sets, pf_sets_convex

areas, indexs, base = [], [], []

# all_results_mp, all_results_meta = [], []
mp_full, meta_full = [], []
mp_full_valid, meta_full_valid, all_valid = [], [], []
pf_mp, pf_meta = [], []

base_mp, base_meta = 0,0

env_name = 'reacher'

fig = plt.figure()
ax_ori = fig.gca(projection='3d')

valid_mp, valid_meta = 0, 0
higher_v_mp, higher_v_meta = 0, 0

# fig = plt.figure()


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

    for j in range(-10, 0, 1):
        if mp_result[j] > meta_result[j]:
            higher_v_mp += 1
        else:
            higher_v_meta += 1

    if i == 0:
        mp_full, meta_full = mp, meta
    else:
        # print(mp_full.shape, mp.shape)
        # print(meta_full.shape, meta.shape)
        mp_full, meta_full = np.concatenate((mp_full, mp), axis=0), np.concatenate((meta_full, meta), axis=0)

    start = len(mp) - 10
    for j in range(start, len(mp)-0, 1):
        sub_mp = np.append(mp[j], 1)
        all_valid.append(sub_mp)
        mp_full_valid.append(sub_mp)
        # if sub_mp[0] > 80:
        if sub_mp[2] > -0.05:
            # mp_full_valid.append(sub_mp)
            # pf_mp, pf_mp_convax = update_pf_sets(pf_mp, sub_mp)
            # ax_ori.scatter(np.asarray(sub_mp)[1], np.asarray(sub_mp)[1], np.asarray(sub_mp)[2], cmap=plt.cm.Spectral, c='r')
            # plt.scatter(np.asarray(sub_mp)[0], np.asarray(sub_mp)[1], c=(1, 0.3, 0.3))
            valid_mp += 1

    start = len(meta) - 10
    for j in range(start, len(meta)-0, 1):
        sub_meta = np.append(meta[j], 0)
        all_valid.append(sub_meta)
        meta_full_valid.append(sub_meta)
        # if sub_meta[0] > 80:
        if sub_meta[2] > -0.05:
            # meta_full_valid.append(sub_meta)
            # pf_meta, pf_meta_convax = update_pf_sets(pf_meta, sub_meta)
            # ax_ori.scatter(np.asarray(sub_meta)[1], np.asarray(sub_meta)[1], np.asarray(sub_meta)[2], cmap=plt.cm.Spectral, c='b')
            # plt.scatter(np.asarray(sub_meta)[0], np.asarray(sub_meta)[1], c=(0.3, 0.3, 1))
            valid_meta += 1

    # pf_mp = update_pf_sets(pf_mp, mp[-1])
    # pf_meta = update_pf_sets(pf_meta, meta[-1])

    # print(len(pf_meta), len(pf_mp), len(pf_meta_convax), len(pf_mp_convax))

print('all policy')
pf_all = pf_meta*1
for sub_mp in pf_mp:
    pf_all, _ = update_pf_sets(pf_all, sub_mp)

pf_meta = np.asarray(pf_meta)
pf_mp = np.asarray(pf_mp)

# pf_meta_convax = np.asarray(pf_meta_convax)
# pf_mp_convax = np.asarray(pf_mp_convax)

pf_all = np.asarray(pf_all)

# print(pf_meta)
# print(pf_mp)

# plt.scatter(np.asarray(pf_meta)[:,0], np.asarray(pf_meta)[:,1], c='b')
# plt.scatter(np.asarray(pf_mp)[:,0], np.asarray(pf_mp)[:,1], c='r')

mp_full_valid = np.asarray(mp_full_valid)
meta_full_valid = np.asarray(meta_full_valid)

min_meta = np.max(-meta_full_valid, axis=0)
min_mp = np.max(-mp_full_valid, axis=0)
min_v = np.maximum(min_meta, min_mp)
min_v = min_v
print(min_meta, min_mp, min_v)


size = 3
hv = hypervolume(np.asarray(-meta_full_valid)[:,0:size])
ref_point = min_v[0:size]
hv_meta = hv.compute(ref_point)

hv = hypervolume(np.asarray(-mp_full_valid)[:,0:size])
ref_point = min_v[0:size]
hv_mp = hv.compute(ref_point)

# print('meta hv', np.log(hv_meta), np.log(hv_mp))
print('meta hv', (hv_meta), (hv_mp))

#-----------------------------------------------------------------
fig = plt.figure()
fig_3d = plt.figure()
ax = fig_3d.gca(projection='3d')

all_valid = np.concatenate((mp_full_valid, meta_full_valid), axis=0)

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

ax.scatter(f_mp[:,0], f_mp[:,1], f_mp[:,2], c='r')
ax.scatter(f_meta[:,0], f_meta[:,1], f_meta[:,2], c='b')
ax.scatter(f_all[:,0], f_all[:,1], f_all[:,2], c='pink')

plt.show()
# print(ndf[0])
#-----------------------------------------------------------------


# # plot_bar(pf_meta, pf_mp)
# # plot_mesh(pf_meta, pf_mp)

# # plot_mesh(pf_meta, pf_mp)


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