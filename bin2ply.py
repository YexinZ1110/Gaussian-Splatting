# import numpy as np

# def read_txt(file_path):
#     # 从 TXT 文件读取点云数据
#     with open(file_path, 'r') as file:
#         points = [list(map(float, line.strip().split())) for line in file.readlines()]
#     return np.array(points)

# def write_ply(points, output_file):
#     # 写入 PLY 文件
#     with open(output_file, 'w') as file:
#         file.write("ply\n")
#         file.write("format ascii 1.0\n")
#         file.write("element vertex {}\n".format(len(points)))
#         file.write("property float x\n")
#         file.write("property float y\n")
#         file.write("property float z\n")
#         file.write("end_header\n")
#         for point in points:
#             file.write("{} {} {}\n".format(point[0], point[1], point[2]))

# # 使用原始字符串来避免路径错误
# input_txt = r"E:\Projects\3d_gaussian\datasets\cone\sparse\0\points3D.txt"
# output_ply = r"E:\Projects\3d_gaussian\datasets\cone\sparse\0\points3D.ply"

# # 读取点云数据
# points = read_txt(input_txt)

# # 写入到 PLY 文件
# write_ply(points, output_ply)

# print("PLY file has been created successfully.")




import numpy as np

def read_txt(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                # 尝试将每一行转换为浮点数列表
                point = list(map(float, line.strip().split()))
                points.append(point)
            except ValueError as e:
                # 打印出错的行和错误信息，继续处理其他行
                print(f"Error processing line: {line}. Error: {e}")
    return points

def write_ply(points, output_file):
    with open(output_file, 'w') as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write("element vertex {}\n".format(len(points)))
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")
        for point in points:
            # 确保每个点都有至少3个坐标值
            if len(point) >= 3:
                file.write("{} {} {}\n".format(point[0], point[1], point[2]))

# 使用原始字符串来避免路径错误
input_txt = r"E:\Projects\3d_gaussian\datasets\cone\sparse\0\points3D.txt"
output_ply = r"E:\Projects\3d_gaussian\datasets\cone\sparse\0\points3D.ply"

# 读取点云数据
points = read_txt(input_txt)

# 写入到 PLY 文件
write_ply(points, output_ply)

print("PLY file has been created successfully.")
