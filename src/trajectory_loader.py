import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd


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
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    results_df = pd.DataFrame({
        'Frame': frames,
        'Intersection over Union': iou_values,
    })
    results_df.to_csv('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/IoU_results.csv',
                      index=False)


def plot_trade_off_results(num_points, iou_values, times):
    plt.figure(figsize=(10, 9))
    plt.plot(num_points, iou_values, 'o-', color='green', label='IoU per #Points')
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
        'Intersection over Union': iou_values,
        'Times': times,
    })
    results_df.to_csv('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/trade_off_results.csv', index=False)