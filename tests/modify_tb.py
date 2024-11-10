from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

ckpt_name = 'DQN_2024-10-15_17-50-48/DQN_UavEnv_f69bf_00000_0_2024-10-15_17-50-51/checkpoint_000050'
rlmodule_ckpt = '/home/wjl/ray_results/Transfer/events.out.tfevents.1729058959.0Pinky0'
input_path = '/home/wjl/ray_results/Transfer/events.out.tfevents.1729058959.0Pinky0'  # 输入需要指定event文件
output_path = '/home/wjl/ray_results/Transfer'  # 输出只需要指定文件夹即可

# 读取需要修改的event文件
ea = event_accumulator.EventAccumulator(input_path)
ea.Reload()
tags = ea.scalars.Keys()  # 获取所有scalar中的keys

# 写入新的文件
# 346 ~ 400
# 1729070128.9584184
# 1729071948.7240856
writer = SummaryWriter(output_path)  # 创建一个SummaryWriter对象
for tag in tags:
    print(tag)
    scalar_list = ea.scalars.Items(tag)

    if tag == 'ray/tune/env_runners/episode_return_mean':  # 修改一下对应的tag即可
        pass

    for scalar in scalar_list:
        if 1729070128.9584184 <= scalar.wall_time and scalar.wall_time <= 1729071948.7240856:
            writer.add_scalar(tag, scalar.value, scalar.step, scalar.wall_time)  # 添加修改后的值到新的event文件中
writer.close()  # 关闭SummaryWriter对象
