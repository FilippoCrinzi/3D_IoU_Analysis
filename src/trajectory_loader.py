import scipy.io as sio
def get_vehicle_position(mat_file=None, frame=None):
    mat = sio.loadmat(mat_file)
    traj_gt_vehicle = mat['Traj_gt_vehicle']
    vehicle_position = traj_gt_vehicle[frame]
    return vehicle_position

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

def plot_iou_results(frames, iou_values, num_points=None):
    import matplotlib.pyplot as plt
    if num_points is not None:
        label_text = f'IoU per Frame (#Points:{num_points})'
    else:
        label_text = 'IoU per Frame'
    plt.plot(frames, iou_values, 'o-', color='blue', label=label_text)
    plt.xlabel('Frame (Tempo)', fontsize=12)
    plt.ylabel('IoU (Intersection over Union)', fontsize=12)
    plt.legend()
    plt.title('Andamento dei Valori di IoU per Frame', fontsize=14)
    plt.grid(True)
    plt.ylim(0, 1)
    plt.tight_layout()