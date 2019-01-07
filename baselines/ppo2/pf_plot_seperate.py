import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

weights = []
for _ in range(30):
    random_w = np.random.rand(5)
    weights.append(random_w)

np.save('/home/xi/workspace/weights_half.npy', weights)

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


def plot_mesh(meta, mp):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    print(meta.shape)
    meta = meta[:, -1, :]
    mp = mp[:, -1, :]
    print(meta.shape)


    # # set_cpy = pf_sets*1
    # # set_cpy.append(np.min(pf_sets, axis=0))
    # mp = np.append(mp, np.min(mp, axis=0))
    # print(mp.shape)
    # hull = ConvexHull(mp[:,0:2])
    # index = hull.vertices        

    # pf_sets_convex = []
    # for i in index:
    #     pf_sets_convex.append(mp[i][0:3])

    # pf_sets_convex = np.asarray(pf_sets_convex)
    # print(pf_sets_convex.shape)
    # # Make data.
    # x = pf_sets_convex[:,0]
    # y = pf_sets_convex[:,1]
    # z = pf_sets_convex[:,2]

    # grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    # grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(grid_x, grid_y, grid_z, cmap=plt.cm.Spectral)
    # plt.show()


    x = mp[:,0]
    y = mp[:,1]
    z = mp[:,2]
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    surf = plt.scatter(x, y,c=(1, 0.2, 0.2),
                        linewidth=0, antialiased=False)
    x = meta[:,0]
    y = meta[:,1]
    z = meta[:,2]
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    surf = plt.scatter(x, y,
                        linewidth=0, antialiased=False)

    plt.show()

def update_pf_sets(pf_sets, sub_target):
    # pf_sets_new = []
    # for pf_set in pf_sets:
    #     if pf_set[3] > 40:
    #         pf_sets_new.append(pf_set)


    # pf_sets = pf_sets_new
    # if len(pf_sets) < 6:
    #     return 0

    max_v = np.max(pf_sets, axis=0)
    min_0 = -130
    pf_sets.append([min_0, -20, max_v[2], -50])
    pf_sets.append([min_0, max_v[1], -5, -50])
    pf_sets.append([max_v[0], -20, -5, -50])
    # pf_sets.append([min_0, -20, -5, max_v[3]])

    # pf_sets.append([min_0, max_v[1], max_v[2], 0])
    # pf_sets.append([max_v[0], max_v[1], -2, 0])
    # pf_sets.append([max_v[0], -15, max_v[2], 0])

    # print(np.min(pf_sets, axis=0))
    is_convax = False
    area = 0
    pf_sets_convex = []
    if len(pf_sets)>6:
        set_cpy = pf_sets*1
        set_cpy.append(np.min(pf_sets, axis=0))
        hull = ConvexHull(np.asarray(set_cpy)[:,0:3])
        index = hull.vertices        

        print(hull.area)
        area = hull.area
        for i in index:
            if i < len(pf_sets):
                pf_sets_convex.append(pf_sets[i])
            # if i == len(pf_sets)-1 and add_candidate:
            #     is_convax = True



    # if len(pf_sets) > 1:
    #     # print(np.asarray(pf_sets)[:,1:3])
    #     plt.clf()
    #     plt.scatter(np.asarray(pf_sets)[:,1], np.asarray(pf_sets)[:,2], c='r')
    #     if pf_sets_convex != []:
    #         plt.scatter(np.asarray(pf_sets_convex)[:,1], np.asarray(pf_sets_convex)[:,2], c='pink')
    #     if sub_target != []:
    #         plt.scatter(sub_target[1], sub_target[2], c='g')
        
    #     plt.scatter(-10, max_v[2], c='blue')
    #     plt.scatter(max_v[1], -200, c='blue')

    #     # pf_sets.append(np.min(pf_sets, axis=0))
    #     # hull = ConvexHull(np.asarray(pf_sets)[:,1:3])
    #     # index = ConvexHull(pf_sets).vertices
    #     # print(index)
    #     # plt.plot(np.asarray(pf_sets)[hull.vertices,1], np.asarray(pf_sets)[hull.vertices,2], 'b--', lw=2)
    #     # plt.plot(np.asarray(pf_sets)[hull.vertices[0],1], np.asarray(pf_sets)[hull.vertices[0],2], 'ro')

    #     # plt.pause(0.01)
    #     plt.show()

        return area

plt.clf()
areas, indexs, base = [], [], []

all_results_mp, all_results_meta = [], []
mp_full, meta_full = [], []

base_l = 0

env_name = 'cheetah'

for i in range(31):
    meta_result = np.load('/home/xi/workspace/exp_models/exp_'+env_name+'/meta_finetune/results_'+str(i)+'.npy')[:,-1]
    mp_result = np.load('/home/xi/workspace/exp_models/exp_'+env_name+'/mp_train/results_'+str(i)+'.npy')[:,-1]

    meta = np.load('/home/xi/workspace/exp_models/exp_'+env_name+'/meta_finetune/results_'+str(i)+'.npy')[:,0:-1]
    mp = np.load('/home/xi/workspace/exp_models/exp_'+env_name+'/mp_train/results_'+str(i)+'.npy')[:,0:-1]

    if i == 0:
        base_l = len(mp_result)
    
    mp_result = mp_result[0:base_l]

    # if mp_result[0] > 0:
    all_results_meta.append(meta_result)
    all_results_mp.append(mp_result)
    
    meta_full.append(meta)
    mp_full.append(mp)

all_results_meta = np.asarray(all_results_meta)
all_results_mp = np.asarray(all_results_mp)
meta_full = np.asarray(meta_full)
mp_full = np.asarray(mp_full)

plot_mesh(meta_full, mp_full)

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