import intersection_over_union as iou
import trade_off as to

path_point_cloud = '/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/point_clouds/PointCloud_traj_argo_50_AV_MercedesGLS580_scans50_s7_h2_5_10_v3_COMfixed_vehicle_time_23_downsampled_gridAverage_step0.60.csv'
point_cloud_dir = '/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/point_clouds/'
trajectories = ('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/trajectories'
                '/traj_argo_50_AV_MercedesGLS580_scans50_s7_h2_5_10_v3_vehicle.mat')
path_mercedes = '/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/models/MercedesGLS580.obj'
num_points = 5000
path_image = '/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/models/PointCloud_traj_argo_50_AV_MercedesGLS580_scans50_s7_h2_5_10_v3_vehicle_time_23.csv'


def main():
    print("Welcome to the IoU tool. Type 'help' for commands.")

    while True:
        command = input("> ").strip()  # Legge l'input e lo normalizza

        if command == "compare":
            print("1. IoU with bounding box")
            print("2. IoU with ellipsoid")
            print("3. IoU with superquadric")
            choice = input("Choose the type of IoU you want to compute: ")
            num_points = int(input("Enter the number of points: "))

            if choice == "1":
                print("Computing IoU for all frames with bounding box...")
                iou.compare_iou(path_mercedes, trajectories, point_cloud_dir, num_points, 'bbox', [0.3, 0.7])
            elif choice == "2":
                print("Computing IoU for all frames with ellipsoid...")
                iou.compare_iou(path_mercedes, trajectories, point_cloud_dir, num_points, 'ellipsoid', [0.3, 0.7])
            elif choice == "3":
                epsilon1 = float(input("Enter the value of epsilon1: "))
                epsilon2 = float(input("Enter the value of epsilon2: "))
                print("Computing IoU for all frames with superquadric...")
                iou.compare_iou(path_mercedes, trajectories, point_cloud_dir, num_points, 'superquadric',
                                [epsilon1, epsilon2])
            else:
                print("Invalid choice. Please choose a number between 1 and 3.")

        elif command == "compute_IoU":
            epsilon1 = float(input("Enter the value of epsilon1: "))
            epsilon2 = float(input("Enter the value of epsilon2: "))
            num_points = int(input("Enter the number of points: "))
            print("Computing IoU for all frames...")
            iou.normal_iou(path_mercedes, trajectories, point_cloud_dir, [epsilon1, epsilon2], num_points)

        elif command == "compute_align_IoU":
            epsilon1 = float(input("Enter the value of epsilon1: "))
            epsilon2 = float(input("Enter the value of epsilon2: "))
            num_points = int(input("Enter the number of points: "))
            print("Computing aligned IoU for all frames...")
            iou.align_iou(path_mercedes, trajectories, point_cloud_dir, [epsilon1, epsilon2], num_points)

        elif command == "trade_off":
            print("1. Bounding box")
            print("2. Ellipsoid")
            print("3. Superquadric")
            shape = input("Choose the shape: ")
            if shape == "1":
                shape = 'bbox'
            elif shape == "2":
                shape = 'ellipsoid'
            elif shape == "3":
                shape = 'superquadric'
            frame = int(input("Enter the frame number: "))
            print("Computing trade-off analysis...")
            to.trade_off(path_mercedes, point_cloud_dir, frame, trajectories, shape)

        elif command == "Epsilon":
            num_points = int(input("Enter the number of points: "))
            print("Start to analyze the best epsilon...")
            iou.superquadric_iou_all_combination(path_mercedes, trajectories, point_cloud_dir, num_points)

        elif command == "visualize":
            print("1. Bounding box")
            print("2. Ellipsoid")
            print("3. Superquadric")
            shape = input("Choose the shape: ")
            if shape == "1":
                shape = 'bbox'
                epsilon = [0.3, 0.7]
            elif shape == "2":
                shape = 'ellipsoid'
                epsilon = [0.3, 0.7]
            elif shape == "3":
                shape = 'superquadric'
                epsilon1 = float(input("Enter the value of epsilon1: "))
                epsilon2 = float(input("Enter the value of epsilon2: "))
                epsilon = [epsilon1, epsilon2]
            frame = int(input("Enter the frame number: "))
            iou.sg.visualize_compare(point_cloud_dir, path_mercedes, trajectories, frame, shape, epsilon)

        elif command == "help":
            print("Available commands:")
            print("- compare: compare the IoU for all frames before and after alignment")
            print("- compute_IoU: compute the IoU for all frames")
            print("- compute_align_IoU: compute the aligned IoU for all frames")
            print("- visualize: visualize the difference between the shape before and after alignment for a specific "
                  "frame")
            print("- trade_off: compute the trade-off analysis")
            print("- epsilon: analyze the best epsilon")
            print("- exit")

        elif command == "exit":
            print("Exiting...")
            break

        else:
            print("Unknown command. Type 'help' for available commands.")


if __name__ == "__main__":
    main()


# iou.compare_iou(path_mercedes, trajectories, point_cloud_dir, num_points, 'ellipsoid', [0.3, 0.7])
# iou.align_iou_for_frame2(path_mercedes, trajectories, point_cloud_dir, 32, num_points, True)
# iou.iou_for_frame(path_mercedes, trajectories, point_cloud_dir, 20, num_points, True)

# iou.iou_ellipsoid_for_frame(path_mercedes, trajectories, point_cloud_dir, 23, num_points, True)

# iou.sg.visualize_compare(point_cloud_dir, path_mercedes, trajectories, 17, 'ellipsoid', [0.3, 0.7])
# iou.align_iou_ellipsoid_for_frame2(path_mercedes, trajectories, point_cloud_dir, 49, num_points, True)

# iou.normal_iou(path_mercedes, trajectories, point_cloud_dir, [0.3, 0.7], num_points)
# iou.align_iou(path_mercedes, trajectories, point_cloud_dir, [0.3,0.7], num_points)

# iou.sg.show_point_cloud(path_image)
# iou.sg.show_ellipsoid(path_point_cloud)

# iou.sg.generate_superquadric_from_point_cloud(path_point_cloud, 0.6, 0.6)
# iou.iou_superquadric_for_frame(path_mercedes, trajectories, point_cloud_dir, 23, num_points, [1, 1], True)

# iou.superquadric_iou_all_frames(path_mercedes, trajectories, point_cloud_dir, num_points, [0.8, 0.7])
# mappa, table = iou.superquadric_iou_all_combination(path_mercedes, trajectories, point_cloud_dir, num_points)
