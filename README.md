# AI-Real-Estate-Agent

本项目以 MIT 协议开源，详见 `LICENSE` 文件。

## 项目整体方案与目标

* **项目总体方案**
  本项目旨在构建一个“端到端可运行”的智能房产推荐系统，能够从房源数据、用户偏好和周边设施三方面出发，自动学习并生成多目标优化的推荐结果。整体分为三层：

  1. **数据层**：加载房源属性（含历史价格）、用户偏好（静态权重与动态自然语言调整）、以及 GeoJSON 格式的周边学校/医院数据，完成清洗、筛选与归一化。
  2. **算法层**：

     * **GNN 表示学习**：将房产与周边设施构建成异构图，利用两层 GCN 提取每个房产节点的高维嵌入，捕捉空间邻里关系；
     * **深度强化学习排序**：基于 DQN，使用房产嵌入作为状态输入，动作为“推荐或不推荐”，多目标奖励（价格、户型、面积、设施距离）按用户权重加权后反馈，训练政策网络；
     * **协同进化优化**：引入 NSGA-II 或 MOEA/D 算法与 RL 协同——进化保证了解的多样性，RL 强化全局搜索定向，两者结合可更充分地逼近真实帕累托前沿。
  3. **交互层**：自然语言解析模块允许用户通过“希望房子便宜但靠近学校”“想要大面积”“学区优先”之类的描述，实时动态调整权重与筛选阈值，并即刻刷新推荐结果。

---

* **研究方向**

  1. **图神经网络在空间推荐系统中的应用**
     * 将异构空间数据（房产＋公共设施）图谱化，实现丰富的邻里影响建模。
  2. **深度强化学习用于多目标排序优化**
     * 探索将 DQN 应用于房产推荐决策过程，摆脱单纯加权求和排序的局限，使策略能够在探索中自适应权衡冲突目标。
  3. **演化算法与深度 RL 的协同搜索**
     * 研究如何将进化多样性与 RL 定向性结合，强化对帕累托前沿的逼近能力。

---

* **核心创新点**

  1. **异构图＋GNN 表征**：首次将房产与学校、医院等设施构建成统一的异构图，通过 GCN 学习出融合地理邻里与属性特征的高质量嵌入。
  2. **多目标 DQN 排序策略**：设计多目标奖励函数，将价格、户型、面积与设施距离等转化为向量化子奖励，并在 DQN 框架下联合优化；支持在线偏好更新后无缝继续训练。
  3. **混合进化-RL 协同优化**：提出“进化＋RL”协同框架：进化算法负责维护解集多样性、避免早熟收敛，RL 提供全局搜索指引，高效逼近非支配解集。
  4. **自然语言驱动的实时交互**：通过规则式关键字解析用户中文偏好描述，无需专业参数设定即可动态调整多目标权重与筛选区间，提升系统易用性和可解释性。

---

* **SCI 论文中体现的创新贡献**
  1. **方法论贡献**：提出了首个将异构 GNN、深度 RL 与进化多目标优化三者融合的房产推荐框架，系统性地验证了各模块对推荐质量和多样性的提升。
  2. **算法创新**：在 DQN 中集成了多目标奖励设计，并结合进化算法的多样性维护策略，实验证明相比传统加权排序或单纯进化算法能更好地逼近真实帕累托前沿。
  3. **系统实现与可复现性**：提供了完整的开源代码包和一键运行脚本，覆盖数据预处理、模型训练、优化搜索与可视化，满足 SCI 对“实验可复现性”的最高标准。
  4. **实验验证**：通过对比实验（单目标加权、纯进化、纯 RL、混合框架）以及帕累托前沿可视化，量化展示了本方法在准确性、多样性和用户满意度上的显著优势。

## 使用说明

### 依赖环境

以下版本在作者环境中验证通过：

- Python 3.10
- PyTorch 2.0.1
- PyTorch-Geometric 2.3.1
- NetworkX 3.1
- Geopy 2.3.0
- Matplotlib 3.7.1

1. 推荐使用提供的一键脚本：`./run.sh`。脚本会自动安装依赖并执行训练，可将自然语言描述作为参数传入：
   `./run.sh --nl "希望房子便宜并靠近学校"`
2. 若已安装好依赖，也可直接运行：`python train.py`
3. 训练完成后可在命令行查看推荐结果或在生成的日志中获取模型指标。

### 复现论文中的对比实验

使用 `experiments.py` 可以分别运行加权排序(`weighted`)、仅进化(`evolution`)、仅强化学习(`rl`)以及混合方法(`hybrid`)四种模式：

```bash
python experiments.py weighted
python experiments.py evolution
python experiments.py rl
python experiments.py hybrid
```
其中 `evolution` 和 `hybrid` 模式会在当前目录生成 `*_pareto.json` 及对应的 PNG 图像，用于展示帕累托前沿。

### 实时交互演示

`interactive_demo.py` 可以读取 `adjust_order.json` 中的自然语言描述或通过 `--nl` 直接指定描述，自动更新偏好并重新输出推荐结果：

```bash
python interactive_demo.py --nl "希望房子更大并靠近医院"
```

### 测试评估

`evaluation.py` 会运行加权排序、无 GNN 的 RL、NSGA-II 以及混合方法，并生成排名指标、覆盖率和训练曲线等结果到 `eval_plots/` 目录：

```bash
python evaluation.py --out eval_plots
```

### English Overview

This repository implements a multi-objective real estate recommender combining GNN, reinforcement learning and evolutionary search. Use `run.sh` or `python train.py` to train the full pipeline. `experiments.py` reproduces ablation studies and exports Pareto front data for visualization. `interactive_demo.py` demonstrates dynamic preference updating via natural language.

## 数据来源与授权

- `facilities.geojson` 来源于 [OpenStreetMap](https://www.openstreetmap.org/)，遵循
  [Open Database License](https://opendatacommons.org/licenses/odbl/)。
- 示例房源数据 `updated_houses_with_price_history.json` 仅用于研究演示，并非真实
  商业数据。

