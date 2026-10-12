# 本机验证记录

截至 2026-10-12，本项目完成以下实际检查：

- 本机 Python：D:/conda_envs/yahui_torch/python.exe。
- torch 2.5.0+cu118，torchvision 0.20.0+cu118，RTX 3050 Laptop GPU。
- 官方 CIFAR-10 压缩包 MD5 为 c58f30108f718f92721af3b95e74349a，已校验；torchvision 对解压数据检查通过。
- 官方数据：50000 张训练图片、10000 张测试图片。
- `python -m unittest discover -s tests -v`：9 项通过，覆盖划分互斥、指标按样本统计、损坏下载拒绝、进程身份和异常退出锁恢复等。
- CPU 真实数据 smoke：`smoke_verified`，已完成。
- CUDA AMP 真实数据 smoke：`smoke_cuda_verified`，已完成。
- 两次 smoke 都仅使用 128 张训练图片、64 张验证图片；保存权重、曲线、混淆矩阵和报告成功，**不可当作正式测试成绩**。
- 另验证了真实优化器更新及权重保存/加载的一致性。
- 正式运行 `resnet18_20261012_093403_514565` 已由后台启动器启动，配置为 30 轮。
- 启动检查时已完成前 2 轮并进入第 3 轮；第 2 轮验证准确率 51.74%，这不是最终测试准确率。best.pt、last.pt、history.csv 和 curves.png 已实际生成。
- 再次调用启动器已确认不会重复创建正在运行的任务。

本记录不宣称正式训练已完成。当前进度以 `SHOW_STATUS.cmd` 为准；完成后，正式指标位于 `evaluation/<实验名称>/metrics.json`。

源码经过独立只读审查。修复了进程 PID 重用误判及遗留锁导致无法启动的问题，并通过回归测试。

下载期间因网络波动使用分段下载和官方 MD5 校验。清理下载临时文件被执行策略拦截，临时文件保留在被 Git 忽略的 data 目录中，不影响训练或上传代码。
