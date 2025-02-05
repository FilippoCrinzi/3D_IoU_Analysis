import re
import scipy.io as sio
import open3d as o3d
import numpy as np


def extract_time_number(filename):
    """ Estrae il numero dopo 'vehicle_time_' in un nome file. """
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