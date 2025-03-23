import re
import scipy.io as sio
import os
import open3d as o3d
import numpy as np
import intersection_over_union as iou
import matplotlib.pyplot as plt
import pandas as pd


def get_vehicle_position(mat_file=None, frame=None):
    mat = sio.loadmat(mat_file)
    traj_gt_vehicle = mat['Traj_gt_vehicle']
    vehicle_position = traj_gt_vehicle[frame]
    return vehicle_position


def extract_time_number(filename):
    match = re.search(r'vehicle_time_(\d+)', filename)
    return int(match.group(1)) if match else float('inf')


def read_file_mat(file_path):
    try:
        # Carica il file .mat
        mat_data = sio.loadmat(file_path)

        # Stampa le chiavi e i contenuti
        print(f"Contenuto del file '{file_path}':\n")
        for key, value in mat_data.items():
            if key.startswith('__'):  # Ignora meta-informazioni
                continue
            print(f"Variabile: {key}")
            print(f"Tipo: {type(value)}")
            print(f"Dimensioni: {getattr(value, 'shape', 'N/A')}")
            print(f"Contenuto: {value}\n")
    except Exception as e:
        print(f"Errore nella lettura del file .mat: {e}")


def sort_directory_files(path_point_cloud_dir):
    point_cloud_files = sorted(
        [f for f in os.listdir(path_point_cloud_dir) if f.endswith('.csv')], key=extract_time_number
    )
    return point_cloud_files


def compute_height_from_bottom(mesh):
    iou.sg.align_mesh_mercedes_to_origin(mesh)
    center = mesh.get_center()
    vertices = np.asarray(mesh.vertices)
    min_z = np.min(vertices[:, 2])
    distance = center[2] - min_z
    bottom_point = np.array([center[0], center[1], min_z])

    # Crea punti visibili
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    center_sphere.paint_uniform_color([1, 0, 0])  # Rosso per il centro
    center_sphere.translate(center)

    bottom_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    bottom_sphere.paint_uniform_color([0, 0, 1])  # Blu per il punto più basso
    bottom_sphere.translate(bottom_point)

    # Mostra tutto
    o3d.visualization.draw_geometries([mesh, center_sphere, bottom_sphere])

    return distance


def plot_iou_results(frames, iou_values_bbox, iou_values_ellipsoid, iou_values_superquadric, num_points):
    plt.plot(frames, iou_values_bbox, 'o-', color='blue', label='IoU per Frame (Bounding Box)', markersize=0, linewidth=1.3)
    plt.plot(frames, iou_values_ellipsoid, 'o-', color='green', label='IoU per Frame (Ellipsoid)', markersize=0, linewidth=1.3)
    plt.plot(frames, iou_values_superquadric, 'o-', color='red', label='IoU per Frame (Superquadric)', markersize=0, linewidth=1.6)
    plt.xlabel('Frame (Tempo)', fontsize=12)
    plt.ylabel('IoU (Intersection over Union)', fontsize=12)
    plt.legend()
    plt.title(f'IoU values per frame ({num_points})', fontsize=14)
    # plt.grid(True)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    results_df = pd.DataFrame({
        'Frame': frames,
        'Intersection over Union bbox': iou_values_bbox,
        'Intersection over Union ellipsoid': iou_values_ellipsoid,
        'Intersection over Union superquadric': iou_values_superquadric,
    })
    results_df.to_csv('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/IoU_results.csv',
                      index=False)
    plt.savefig('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/IoU_graph.png', dpi=500)


def plot_iou_align_results(frames, iou_values_bbox, iou_values_ellipsoid, iou_values_superquadric, num_points):
    plt.plot(frames, iou_values_bbox, 'o-', color='blue', label='IoU per Frame (Bounding Box)', markersize=0, linewidth=1.3)
    plt.plot(frames, iou_values_ellipsoid, 'o-', color='green', label='IoU per Frame (Ellipsoid)', markersize=0, linewidth=1.3)
    plt.plot(frames, iou_values_superquadric, 'o-', color='red', label='IoU per Frame (Superquadric)', markersize=0, linewidth=1.6)
    plt.xlabel('Frame (Tempo)', fontsize=12)
    plt.ylabel('IoU (Intersection over Union)', fontsize=12)
    plt.legend()
    plt.title(f'Align IoU values per Frame ({num_points})', fontsize=14)
    # plt.grid(True)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    results_df = pd.DataFrame({
        'Frame': frames,
        'Intersection over Union bbox': iou_values_bbox,
        'Intersection over Union ellipsoid': iou_values_ellipsoid,
        'Intersection over Union superquadric': iou_values_superquadric,
    })
    results_df.to_csv('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/IoU_align_results.csv',
                      index=False)
    plt.savefig('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/IoU_align_graph.png', dpi=500)


def plot_trade_off_results(num_points, iou_values, iou_align_values, times, times_align):
    plt.figure(figsize=(10, 9))
    plt.plot(num_points, iou_values, 'o-', color='green', label='IoU per #Points', markersize=6, linewidth=1.6)
    plt.plot(num_points, iou_align_values, 'o-', color='blue', label='Align IoU per #Points', markersize=6, linewidth=1.6)
    plt.xlabel('Numero di Punti', fontsize=12)
    plt.ylabel('IoU (Intersection over Union)', fontsize=12)
    plt.legend()
    plt.title('Trade-Off', fontsize=14)
    plt.grid(True)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    plt.show()
    results_df = pd.DataFrame({
        'Points': num_points,
        'IoU': iou_values,
        'Times': times,
        'Align IoU': iou_align_values,
        'Times Align': times_align,
    })
    results_df.to_csv('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/trade_off_results.csv', index=False)
    plt.savefig('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/trade_off.png', dpi=400)


def plot_trade_off_time(num_points, times, times_align):
    plt.figure(figsize=(10, 9))
    plt.plot(num_points, times, 'o-', color='green', label='Time per #Points', markersize=6, linewidth=1.6)
    plt.plot(num_points, times_align, 'o-', color='blue', label='Align Time per #Points', markersize=6, linewidth=1.6)
    plt.xlabel('Numero di Punti', fontsize=12)
    plt.ylabel('Tempo (s)', fontsize=12)
    plt.legend()
    plt.title('Trade-Off', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    results_df = pd.DataFrame({
        'Points': num_points,
        'Times': times,
        'Times Align': times_align,
    })
    results_df.to_csv('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/trade_off_time.csv', index=False)
    plt.savefig('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/trade_off_time.png', dpi=400)


def plot_compare_results(frames, iou_values, iou_align_values, num_points, shape, epsilon):
    plt.plot(frames, iou_values, 'o-', color='blue', label='IoU per Frame', markersize=0, linewidth=1.3)
    plt.plot(frames, iou_align_values, 'o-', color='green', label='Align IoU per Frame', markersize=0, linewidth=1.3)
    plt.xlabel('Frame (Tempo)', fontsize=12)
    plt.ylabel('IoU (Intersection over Union)', fontsize=12)
    plt.legend()
    if shape == 'superquadric':
        plt.title(f'IoU values per frame({shape},{num_points},{epsilon})', fontsize=14)
    else:
        plt.title(f'IoU values per frame({shape},{num_points})', fontsize=14)
    plt.grid(True)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    results_df = pd.DataFrame({
        'Frame': frames,
        'IoU': iou_values,
        'Align_IoU': iou_align_values,
    })
    results_df.to_csv('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/compare_results.csv',
                      index=False)
    plt.savefig('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/compare_graph.png', dpi=500)


def sign_power(base, exp):
    """ Implementa x^e mantenendo il segno di x per supportare e < 1 """
    return np.sign(base) * (np.abs(base) ** exp)


def compute_T_matrix(yaw, pitch, roll):
    """ Calcola la matrice di rotazione T a partire dai valori di yaw, pitch e roll """
    T = np.zeros((3, 3))
    T[0, 0] = np.cos(pitch) * np.cos(roll)
    T[0, 1] = np.cos(pitch) * np.sin(roll)
    T[0, 2] = -np.sin(pitch)
    T[1, 0] = np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)
    T[1, 1] = np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll)
    T[1, 2] = np.sin(yaw) * np.cos(pitch)
    T[2, 0] = np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)
    T[2, 1] = np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll)
    T[2, 2] = np.cos(yaw) * np.cos(pitch)
    return T

'''
ellips = [0.6255230125523012,
0.6457461645746164,
0.6573816155988857,
0.6325947105075054,
0.6577249575551782,
0.6328097731239093,
0.6540922869653082,
0.629979035639413,
0.6263033972418433,
0.6324029952348537,
0.6306680512287989,
0.6121873899260303,
0.6208547008547008,
0.6317962835512733,
0.6402170227195659,
0.6117850953206239,
0.5882162512139851,
0.6061312895701433,
0.5983908816627557,
0.5839320705421294,
0.5913477537437604,
0.5641971169963125,
0.5905377808032676,
0.6186237589866485,
0.6267910696434522,
0.6475437993816557,
0.6337290502793296,
0.6409556313993174,
0.6532285616672361,
0.6606701940035273,
0.6227461858529819,
0.6200219218122032,
0.6054519368723099,
0.6029082774049217,
0.6079016026835632,
0.5463102409638554,
0.5700227100681302,
0.5411808950733359,
0.5246159905474597,
0.5242290748898678,
0.5199374511336982,
0.5398967844382692,
0.5194706994328923,
0.4980237154150198,
0.5190270694389957,
0.49979449239621865,
0.5090180360721442,
0.5289356535815459,
0.5174230145867099,
0.5244532803180915]
ellips_align = [0.6784534038334434,
0.6620291822192059,
0.6661146074859224,
0.6750334672021419,
0.6672297297297297,
0.6732203389830509,
0.6842991913746631,
0.6733108108108108,
0.6792707629979743,
0.6954912516823688,
0.6843546730571722,
0.6701316701316701,
0.6991680532445923,
0.6909522195865808,
0.6738255033557047,
0.6773120425815037,
0.6757915567282322,
0.6959742351046699,
0.6783854166666666,
0.6918936035465485,
0.6802830492119653,
0.6744338693797177,
0.7068327162571525,
0.670549084858569,
0.6810957051065268,
0.657566565554432,
0.6656441717791411,
0.6758525663107131,
0.6718106995884774,
0.6620507748104187,
0.6819112627986348,
0.65732832251452,
0.6471186440677966,
0.6624472573839663,
0.625086385625432,
0.6210078069552875,
0.5956691515796947,
0.5626589175444969,
0.5581140350877193,
0.5518532384874579,
0.5406386066763426,
0.5373520710059172,
0.5462027684249906,
0.5323599052880821,
0.5462998837659822,
0.5415993537964459,
0.5435897435897435,
0.5418679549114331,
0.5322391559202814,
0.5357570914902118]

superq =[0.6571128975882559,
0.6303200844178685,
0.6406864497676081,
0.6742424242424242,
0.6807289908876138,
0.6787938380858735,
0.6589381720430108,
0.6374541819393535,
0.6526926797807159,
0.6525285481239804,
0.6484076433121019,
0.6538705583756346,
0.6618250710451531,
0.64801657785672,
0.6389922702547953,
0.6431034482758621,
0.6509333333333334,
0.6612689142553756,
0.6439636752136753,
0.642838907327036,
0.6298444503031901,
0.623023074928701,
0.6584080126515551,
0.676019057702488,
0.6531776208582292,
0.6703387496557422,
0.6821509009009009,
0.7017898383371824,
0.6716284608514439,
0.6867574616053318,
0.6782115869017632,
0.6558375634517767,
0.6649642492339122,
0.6701766304347826,
0.6782367233599444,
0.6261612783351914,
0.6552114803625377,
0.6032995875515561,
0.590819153146023,
0.568076923076923,
0.5748865355521936,
0.5646032405484005,
0.5838368580060423,
0.5849772382397572,
0.5887463798096815,
0.5595041322314049,
0.5721271393643031,
0.5682656826568265,
0.5842650103519669,
0.6012320328542095]

superq_align = [0.7436887436887437,
0.7404103479036575,
0.7387657279808268,
0.75541438059486,
0.7603550295857988,
0.7697555429221148,
0.7583684950773558,
0.7519138077686419,
0.7489469250210615,
0.7474664475486168,
0.7493962972900456,
0.7537648612945839,
0.7437466737626397,
0.7649495210975925,
0.7646017699115044,
0.762338648443432,
0.7668118045476536,
0.7558569667077681,
0.7371044106653376,
0.7792303897385299,
0.752153581097711,
0.7659522049765952,
0.7591981730525248,
0.7324654622741764,
0.7129553546973432,
0.7072833010246469,
0.7184409540430483,
0.7268505079825834,
0.7089774078478003,
0.7283220720720721,
0.7599293909973521,
0.7531743573861877,
0.7425960932577189,
0.7584138620459847,
0.7173427020121367,
0.6896186440677966,
0.6912607449856734,
0.6951646811492642,
0.6534763313609467,
0.6005961251862891,
0.635664591724643,
0.6042477876106195,
0.6373825018076645,
0.6287110108981586,
0.628032345013477,
0.57524557956778,
0.6298904538341158,
0.6087295401402962,
0.6297169811320755,
0.6403050983540747]

percentuali = []
for i in range(len(superq)):
    percentuale = (superq_align[i] - superq[i]) / superq[i] * 100
    percentuali.append(percentuale)
media = np.mean(percentuali)
print(media)
'''
