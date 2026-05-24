# 终端与 Shell

> 终端是 AI 工程师的栖息地。在这里打好根基。

**类型：** 学习
**语言：** --
**前置要求：** 阶段 0，第 01 课
**时间：** 约 35 分钟

## 学习目标

- 使用管道、重定向和 `grep` 从命令行过滤和处理训练日志
- 创建带多个面板的持久 tmux 会话，同时运行训练和 GPU 监控
- 使用 `htop`、`nvtop` 和 `nvidia-smi` 监控系统和 GPU 资源
- 使用 SSH、`scp` 和 `rsync` 在本地和远程机器之间传输文件

## 问题

你在终端里花的时间会超过在任何编辑器里的时间：训练运行、GPU 监控、日志追踪、远程 SSH 会话、环境管理。每个 AI 工作流都会碰到 Shell。在这里慢，到处都会慢。

本课涵盖 AI 工作所需的终端技能。不讲 Unix 历史，不深挖 Bash 脚本，只讲你需要的。

## 概念

```mermaid
graph TD
    subgraph tmux["tmux 会话：training"]
        subgraph top["上行"]
            P1["面板 1：训练运行<br/>python train.py<br/>Epoch 12/100 ..."]
            P2["面板 2：GPU 监控<br/>watch -n1 nvidia-smi<br/>GPU: 78% | Mem: 14/24G"]
        end
        P3["面板 3：日志 + 实验<br/>tail -f logs/train.log | grep loss"]
    end
```

三件事同时运行，一个终端。你可以断开连接，回家，SSH 重新连接，再挂载回来。训练一直在跑。

## 动手实现

### 第一步：了解你的 Shell

查看当前运行的是哪种 Shell：

```bash
echo $SHELL
```

大多数系统使用 `bash` 或 `zsh`，两者都可以。本课程的命令在两者中都能运行。

基础要点：

```bash
# 导航
cd ~/projects/ai-engineering-from-scratch
pwd
ls -la

# 历史搜索（最有用的快捷键）
# Ctrl+R 然后输入之前命令的一部分
# 再次按 Ctrl+R 可以在匹配项间切换

# 清除终端
clear   # 或 Ctrl+L

# 取消正在运行的命令
# Ctrl+C

# 暂停正在运行的命令（用 fg 恢复）
# Ctrl+Z
```

### 第二步：管道与重定向

管道将命令连接在一起。这是处理日志、过滤输出和链接工具的方式，你会频繁用到。

```bash
# 统计训练日志中 "loss" 出现的次数
cat train.log | grep "loss" | wc -l

# 从训练输出中提取 loss 值
grep "loss:" train.log | awk '{print $NF}' > losses.txt

# 实时查看日志文件更新，过滤错误信息
tail -f train.log | grep --line-buffered "ERROR"

# 按最终准确率排序实验结果
grep "final_accuracy" results/*.log | sort -t= -k2 -n -r

# 将 stdout 和 stderr 分别重定向到不同文件
python train.py > output.log 2> errors.log

# 将两者都重定向到同一文件
python train.py > train_full.log 2>&1
```

你需要掌握的三种重定向：

| 符号 | 作用 |
|------|------|
| `>` | 将 stdout 写入文件（覆盖） |
| `>>` | 将 stdout 追加到文件 |
| `2>` | 将 stderr 写入文件 |
| `2>&1` | 将 stderr 发送到与 stdout 相同的目标 |
| `\|` | 将一个命令的 stdout 作为下一个命令的 stdin |

### 第三步：后台进程

训练运行需要数小时。你不想一直开着终端等待。

```bash
# 在后台运行（输出仍然显示在终端）
python train.py &

# 在后台运行，不受终端关闭影响
nohup python train.py > train.log 2>&1 &

# 查看后台运行的任务
jobs
ps aux | grep train.py

# 将后台任务调回前台
fg %1

# 终止后台进程
kill %1
# 或者找到 PID 后终止
kill $(pgrep -f "train.py")
```

`&`、`nohup` 和 `screen`/`tmux` 的区别：

| 方法 | 终端关闭后继续？ | 可以重新连接？ |
|------|---------------|--------------|
| `command &` | 否 | 否 |
| `nohup command &` | 是 | 否（查看日志文件） |
| `screen` / `tmux` | 是 | 是 |

对于任何超过几分钟的任务，使用 tmux。

### 第四步：tmux

tmux 可以创建带多个面板的持久终端会话。这是管理训练运行最有用的工具。

```bash
# 安装
# macOS
brew install tmux
# Ubuntu
sudo apt install tmux

# 创建命名会话
tmux new -s training

# 水平分割
# Ctrl+B 然后 "

# 垂直分割
# Ctrl+B 然后 %

# 在面板间导航
# Ctrl+B 然后方向键

# 断开连接（会话继续运行）
# Ctrl+B 然后 d

# 重新连接
tmux attach -t training

# 列出所有会话
tmux ls

# 结束会话
tmux kill-session -t training
```

典型的 AI 工作流会话：

```bash
tmux new -s train

# 面板 1：开始训练
python train.py --epochs 100 --lr 1e-4

# Ctrl+B, " 分割，然后运行 GPU 监控
watch -n1 nvidia-smi

# Ctrl+B, % 垂直分割，查看日志
tail -f logs/experiment.log

# 用 Ctrl+B, d 断开连接
# SSH 退出，去喝咖啡，回来
# tmux attach -t train
```

### 第五步：使用 htop 和 nvtop 监控资源

```bash
# 系统进程（比 top 更好）
htop

# GPU 进程（如果有 NVIDIA GPU）
# 安装：sudo apt install nvtop（Ubuntu）或 brew install nvtop（macOS）
nvtop

# 快速查看 GPU，无需 nvtop
nvidia-smi

# 每秒更新 GPU 使用情况
watch -n1 nvidia-smi

# 查看哪些进程在使用 GPU
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
```

`htop` 常用快捷键：
- `F6` 或 `>` 按列排序（按内存排序找内存泄漏）
- `F5` 切换树状视图（查看子进程）
- `F9` 终止进程
- `/` 搜索进程名

### 第六步：用 SSH 连接远程 GPU 服务器

租用云 GPU（Lambda、RunPod、Vast.ai）时，通过 SSH 连接。

```bash
# 基本连接
ssh user@gpu-box-ip

# 使用指定密钥
ssh -i ~/.ssh/my_gpu_key user@gpu-box-ip

# 复制文件到远程
scp model.pt user@gpu-box-ip:~/models/

# 从远程复制文件
scp user@gpu-box-ip:~/results/metrics.json ./

# 同步整个目录（适合文件较多的情况，速度更快）
rsync -avz ./data/ user@gpu-box-ip:~/data/

# 端口转发（本地访问远程 Jupyter/TensorBoard）
ssh -L 8888:localhost:8888 user@gpu-box-ip
# 然后在浏览器中打开 localhost:8888

# SSH 配置（便于快速连接）
# 添加到 ~/.ssh/config：
# Host gpu
#     HostName 192.168.1.100
#     User ubuntu
#     IdentityFile ~/.ssh/gpu_key
#
# 然后直接：
# ssh gpu
```

### 第七步：AI 工作实用别名

将以下内容添加到 `~/.bashrc` 或 `~/.zshrc`：

```bash
source phases/00-setup-and-tooling/10-terminal-and-shell/code/shell_aliases.sh
```

或复制你需要的别名。关键别名：

```bash
# 一眼看清 GPU 状态
alias gpu='nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'

# 终止所有 Python 训练进程
alias killtraining='pkill -f "python.*train"'

# 快速激活虚拟环境
alias ae='source .venv/bin/activate'

# 监控训练 loss
alias watchloss='tail -f logs/*.log | grep --line-buffered "loss"'
```

完整别名列表见 `code/shell_aliases.sh`。

### 第八步：常见 AI 终端操作模式

这些在实践中会反复出现：

```bash
# 运行训练、记录所有输出、完成后发通知
python train.py 2>&1 | tee train.log; echo "DONE" | mail -s "Training complete" you@email.com

# 并排比较两个实验日志
diff <(grep "accuracy" exp1.log) <(grep "accuracy" exp2.log)

# 找到最大的模型文件（清理磁盘空间）
find . -name "*.pt" -o -name "*.safetensors" | xargs du -h | sort -rh | head -20

# 从 Hugging Face 下载模型
wget https://huggingface.co/model/resolve/main/model.safetensors

# 解压数据集
tar xzf dataset.tar.gz -C ./data/

# 统计所有 Python 文件的行数（了解项目规模）
find . -name "*.py" | xargs wc -l | tail -1

# 检查磁盘空间（训练数据很快填满磁盘）
df -h
du -sh ./data/*

# 训练前检查环境变量
env | grep -i cuda
env | grep -i torch
```

## 实际使用

课程中各工具的使用时机：

| 工具 | 使用场景 |
|------|---------|
| tmux | 每次训练运行（阶段 3 及以后） |
| `tail -f` + `grep` | 监控训练日志 |
| `nohup` / `&` | 快速后台任务 |
| `htop` / `nvtop` | 调试训练速度慢、显存溢出问题 |
| SSH + `rsync` | 在云 GPU 上工作 |
| 管道 + 重定向 | 处理实验结果 |
| 别名 | 节省重复命令的时间 |

## 练习

1. 安装 tmux，创建一个有三个面板的会话，在一个面板中运行 `htop`，另一个运行 `watch -n1 date`，第三个运行 Python 脚本。断开连接后重新连接。
2. 将 `code/shell_aliases.sh` 中的别名添加到 Shell 配置中，并用 `source ~/.zshrc`（或 `~/.bashrc`）重新加载。
3. 用 `for i in $(seq 1 100); do echo "epoch $i loss: $(echo "scale=4; 1/$i" | bc)"; sleep 0.1; done > fake_train.log` 创建一个假训练日志，然后使用 `grep`、`tail` 和 `awk` 提取 loss 值。
4. 为你有访问权限的服务器设置 SSH 配置（或使用 `localhost` 练习语法）。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|----------------------|
| Shell | "终端" | 解释你命令的程序（bash、zsh、fish） |
| tmux | "终端复用器" | 在一个窗口内运行多个终端会话并支持断开/重连的程序 |
| 管道（Pipe）| "竖线那个" | `\|` 操作符，将一个命令的输出作为另一个命令的输入 |
| PID | "进程 ID" | 分配给每个运行进程的唯一编号，用于监控或终止进程 |
| nohup | "不挂断" | 运行一个不受挂断信号影响的命令，关闭终端不会终止它 |
| SSH | "连接到服务器" | 安全 Shell，一种在远程机器上运行命令的加密协议 |
