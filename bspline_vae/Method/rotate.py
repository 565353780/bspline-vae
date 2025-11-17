import torch


def random_rotation_matrices(batch_size):
    """生成 batch_size 个随机旋转矩阵 [batch_size,3,3]"""
    rand_quat = torch.randn(batch_size, 4)
    rand_quat = rand_quat / rand_quat.norm(dim=1, keepdim=True)  # 归一化
    w, x, y, z = rand_quat[:,0], rand_quat[:,1], rand_quat[:,2], rand_quat[:,3]
    # 四元数转旋转矩阵
    R = torch.zeros(batch_size, 3, 3)
    R[:,0,0] = 1 - 2*(y**2 + z**2)
    R[:,0,1] = 2*(x*y - z*w)
    R[:,0,2] = 2*(x*z + y*w)
    R[:,1,0] = 2*(x*y + z*w)
    R[:,1,1] = 1 - 2*(x**2 + z**2)
    R[:,1,2] = 2*(y*z - x*w)
    R[:,2,0] = 2*(x*z - y*w)
    R[:,2,1] = 2*(y*z + x*w)
    R[:,2,2] = 1 - 2*(x**2 + y**2)
    return R
