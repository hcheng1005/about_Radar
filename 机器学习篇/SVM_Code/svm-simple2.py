import matplotlib.pyplot as plt
import numpy as np
import random

def load_data_set(file_name):
    """ Load data from file.
    
    Args:
        file_name (str): The name of the file from which to load the data.
        
    Returns:
        tuple: Tuple containing the data matrix and label matrix.
    """
    data_matrix = []
    label_matrix = []
    with open(file_name) as file:
        for line in file.readlines():
            line_array = line.strip().split('\t')
            data_matrix.append([float(line_array[0]), float(line_array[1])])
            label_matrix.append(float(line_array[2]))
    return data_matrix, label_matrix

def select_random_j(i, m):
    """ Randomly select index j, different from i.
    
    Args:
        i (int): Index of the first alpha.
        m (int): Total number of alphas.
        
    Returns:
        int: The index j.
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

def clip_alpha(aj, high, low):
    """ Clip alpha value to be within a specified range.
    
    Args:
        aj (float): The alpha value to clip.
        high (float): The high limit for alpha.
        low (float): The low limit for alpha.
        
    Returns:
        float: The clipped alpha value.
    """
    return max(low, min(aj, high))

def show_data_set(data_matrix, label_matrix):
    """ Visualize the data set with a scatter plot.
    
    Args:
        data_matrix (list): List of data points.
        label_matrix (list): List of labels corresponding to the data points.
    """
    data_plus = [data_matrix[i] for i in range(len(data_matrix)) if label_matrix[i] > 0]
    data_minus = [data_matrix[i] for i in range(len(data_matrix)) if label_matrix[i] <= 0]
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(data_plus_np[:, 0], data_plus_np[:, 1], s=30, alpha=0.7, label='Positive')
    plt.scatter(data_minus_np[:, 0], data_minus_np[:, 1], s=30, alpha=0.7, label='Negative')
    plt.legend()
    # plt.show()
def smo_simple(data_matrix, class_labels, C, tolerance, max_iter):
    """ 简化版的序列最小优化（SMO）算法，用于训练支持向量机。
    
    参数:
        data_matrix (list): 数据点矩阵。
        class_labels (list): 数据点的标签。
        C (float): 正则化参数。
        tolerance (float): 停止准则的容忍度。
        max_iter (int): 最大迭代次数。
        
    返回:
        tuple: 包含模型参数b和alphas的元组。
    """
    data_matrix = np.mat(data_matrix)
    label_mat = np.mat(class_labels).transpose()
    b = 0
    m, n = np.shape(data_matrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter_num = 0
    while iter_num < max_iter:
        alpha_pairs_changed = 0
        for i in range(m):
            # 计算第i个数据点的预测值和误差
            fXi = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[i, :].T)) + b
            Ei = fXi - float(label_mat[i])
            # 检查是否违反KKT条件
            if ((label_mat[i] * Ei < -tolerance) and (alphas[i] < C)) or ((label_mat[i] * Ei > tolerance) and (alphas[i] > 0)):
                j = select_random_j(i, m)  # 随机选择第j个数据点
                fXj = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T)) + b
                Ej = fXj - float(label_mat[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                # 计算剪辑的边界L和H
                if label_mat[i] != label_mat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[i] + alphas[j])
                if L == H:
                    continue
                # 计算eta
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - data_matrix[i, :] * data_matrix[i, :].T - data_matrix[j, :] * data_matrix[j, :].T
                if eta >= 0:
                    continue
                # 更新alpha_j
                alphas[j] -= label_mat[j] * (Ei - Ej) / eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    continue
                # 更新alpha_i
                alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])
                # 更新b
                b1 = b - Ei - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[i, :].T - label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[i, :] * data_matrix[j, :].T
                b2 = b - Ej - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[j, :].T - label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[j, :].T
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
        if alpha_pairs_changed == 0:
            iter_num += 1
        else:
            iter_num = 0
    return b, alphas

def calculate_w(data_matrix, label_matrix, alphas):
    """ Calculate the weight vector w for the SVM.
    
    Args:
        data_matrix (list): List of data points.
        label_matrix (list): List of labels.
        alphas (np.matrix): Alphas values from the SMO algorithm.
        
    Returns:
        np.array: The calculated weight vector w.
    """
    data_matrix = np.array(data_matrix)
    label_matrix = np.array(label_matrix)
    w = np.dot((np.tile(label_matrix.reshape(-1, 1), (1, 2)) * data_matrix).T, alphas).flatten()
    return w

def plot_decision_boundary(data_matrix, label_matrix, w, b):
    """ Plot the decision boundary of the SVM along with the data points.
    
    Args:
        data_matrix (list): List of data points.
        label_matrix (list): List of labels.
        w (np.array): Weight vector.
        b (float): Bias term.
    """
    show_data_set(data_matrix, label_matrix)

    # Create an array of x values for plotting the decision boundary.
    x_values = np.linspace(min(data_matrix[:, 0]), max(data_matrix[:, 0]), 100)
    # Calculate the corresponding y values using the equation of the hyperplane.
    y_values = (-b - w[0, 0] * x_values) / w[0, 1]
    plt.plot(x_values, y_values.T, 'r-')
    plt.show()
    
if __name__ == '__main__':
    # Load data from file
    data_matrix, label_matrix = load_data_set('testSet.txt')
    
    # Run the SMO algorithm to find the optimal alphas and b
    b, alphas = smo_simple(data_matrix, label_matrix, 0.6, 0.001, 40)
    
    # Calculate the weight vector w using the alphas
    w = calculate_w(data_matrix, label_matrix, alphas)
    
    # Visualize the results including the decision boundary
    plot_decision_boundary(np.array(data_matrix), np.array(label_matrix), w, b)