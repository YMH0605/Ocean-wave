# 项目交接报告

**日期**：2026-08-17
**用途**：在新对话中继续这个项目。读完这份就能直接开始工作，不需要回看旧对话。

---

# 0. 新对话要做的第一件事

**不需要上传任何文件。** 代码、数据、结果全在
`C:\Users\Minghao\Desktop\Ocean Wave`，新对话可以直接读。

只需要说一句：

> 读一下 `C:\Users\Minghao\Desktop\Ocean Wave\HANDOVER.md`，我们继续这个项目。

**唯一在项目外的东西**：原始论文 PDF 在
`C:\Users\Minghao\Downloads\1-s2.0-S0029801825019535-main.pdf`（如果需要再查原文）。

---

# 1. 项目是什么

复现 **Ma, Ding, Ai, Liu & Bu (2025)**, *"Prediction of wave parameters under
extreme sea conditions using a 3D U-Net deep learning model"*,
Ocean Engineering 340:122269 —— 从西北太平洋搬到 **北卡 Cape Hatteras / Outer Banks**。

**复现只是脚手架，不是终点。** 最终方向是
**physics-informed neural operator (PINO)**。先做 3D U-Net 是为了拿到一套干净
可信的数据、基线和评估框架，让后续任何架构都有东西可比。

**当前定位**：投**会议论文**（非顶会）。用户明确说这个范围够了。

---

# 2. 关键决策与理由（这些不在代码里）

| 决策 | 理由 |
|---|---|
| **不把研究 scope 定在台风** | 见 §4.1。Outer Banks 79% 的极端事件是温带风暴，冬季零台风 |
| **训练用连续记录，不做事件切片** | 论文的方法在这里只覆盖 3% 的记录 |
| **时段 2000–2025**（不是 1979 起） | 1991 年前 ERA5 波浪场缺乏卫星高度计约束，质量存疑。且和论文训练期 2000–2019 对齐 |
| **不加 MP2** | 用户认为会让工作和原论文过于相似。**我最初建议加，后来收回** |
| **不追遥感期刊** | 卫星在我们的设计里是配角，强行推到中心要重做方法 |
| **浮标只做验证，不进训练** | 13 个点 vs 1536 个格点，信号占比可忽略；且会失去唯一的独立参照 |
| **不和 persistence/climatology 对比** | 用户认为它们不算真正的 baseline。**notebook 里已全部移除** |
| **不跑 WW3/SWAN** | 会议论文可以不做，期刊会被问 |

---

# 3. 环境

```
Python      D:\Anaconda\envs\oceanwave\python.exe   (Python 3.12)
            ⚠️ 不能用 base 环境，Anaconda 的 MKL 和 PyTorch 的 OpenMP 冲突
GPU         RTX 5090 (32 GB), torch 2.11.0+cu128, sm_120
Jupyter     内核名 "Python (oceanwave)" 已注册
conda       PowerShell 未初始化。跑 `D:\Anaconda\Scripts\conda.exe init powershell`
            重开终端后 `conda activate oceanwave`，之后 `python` 直接可用
CDS         凭证在 %USERPROFILE%\.cdsapirc（token 是 UUID 格式，2024 年后的新版）
```

**仓库**：https://github.com/YMH0605/Ocean-wave **（Public）**

> ⚠️ 用户已把 README 清空（移除了对论文的批评和未发表的发现）。
> **新对话不要往 README 里加研究结论。**
> 中文教学大纲 `TEACHING_GUIDE.md` 已 gitignore，不上传（学生是 ABC，且用户不想让他看到）。

---

# 4. 已完成的工作与全部结果

## 4.1 HURDAT2 风暴目录分析（已归档）

**结论：论文的"事件切片"方法在这个区域失效。**

```
1979–2024，Hatteras 500 km 内：148 场风暴（3.2/年）
事件窗口覆盖率：3.0%  → 论文方法会丢掉 97% 的训练数据
3 级以上飓风：46 年里只有 5 场
最近点仍是热带气旋：66%（22% 已温带转换）
12/2/3/4 月：零事件
```

**改用波高场定义极端事件**（P99 阈值、持续 ≥12h），2000–2025 共 **76 场**：

```
温带风暴 60 场（78.9%），共 1,523 小时
热带气旋 16 场（21.1%），共   525 小时
12–3 月的 58 场全部是温带风暴，热带气旋零场
两类平均强度几乎相同（ET 9.15 m vs TC 9.14 m），差别只在频次
26 年最大的一场浪 12.00 m 出现在 2014-03-27，是东北风暴，HURDAT2 查不到
```

**这部分已移到 `_archive/extreme_analysis/`**（用户要求，避免学生看乱）。
里面有 README 说明如何恢复。**写论文时这是核心贡献之一，需要移回来。**

## 4.2 数据

```
ERA5  2000–2025，逐小时，227,928 步，零时间断点
域    28.5–44.0°N, 83.5–60.0°W  →  32 × 48 @ 0.5°（每格约 56 × 46 km）
      32 和 48 都能被 8 整除（U-Net 三次池化要求）
陆地占比 27.8%
通道  swh, mp2, pp1d, mwd_sin, mwd_cos, u10, v10  (7 个) + 陆地掩膜 = 8
划分  训练 2000–2020（除验证年）/ 验证 2004,2009,2014,2019 / 测试 2021–2025
下载  156 个两月块（CDS 成本上限 121,000，整年 630,720 会被拒），耗时 31 小时
```

**1979–1997 的数据已下载但归档**在 `data/raw/era5/archive_pre2000/`，
可用于"数据量是否有用"的消融实验。

## 4.3 与论文的所有差异

| 论文 | 我们 | 影响方向 |
|---|---|---|
| 无 persistence 基线 | 补上了（后应用户要求从 notebook 移除） | — |
| 波向 0–360 直接 min-max | 拆 sin/cos | 改进 |
| 陆地填 0 且进 loss | 掩膜通道 + masked loss | 改进 |
| 随机 80/20 划分 | 按年份划分 | 我们更严格 |
| 归一化统计用全记录 | 只用训练年份 | 我们更严格 |
| 只有 1 小时时效 | 1/6/24/48 | 扩展 |
| 未提训练稳定性 | 梯度裁剪 + warmup | 必需（无裁剪时发散过） |
| **预测 SWH 时输入不含 SWH** | 两种模式都实现 | **单此一项影响 +83%** |

**指标口径**：我们只算海洋点，论文含陆地。实测差异 **28%**——
按论文口径我们的数字会更好看（0.021 → 0.015），**所以我们的报告是保守的**。

## 4.4 模型结果

**+1 小时，测试集 2021–2025（ERA5 为参照）**

```
model                        params      lr  base    MAE   RMSE    bias      R²
3D U-Net                  5,920,353   0.001   32   0.020  0.037  +0.005   0.999
ConvLSTM                    461,377   0.001   32   0.024  0.038  -0.001   0.999
3D CNN                    3,913,857   0.001   32   0.030  0.049  +0.000   0.998
3D U-Net (论文通道)        5,920,353   0.001   32   0.039  0.057  +0.009   0.997
```

**+24 小时**

```
3D U-Net   0.332    3D CNN   0.334    ConvLSTM   0.402
```

**排名随时效反转**：ConvLSTM 从第 2 掉到最后；3D CNN 从最后升到并列第一。
skip connection 在 1h 值三分之一的误差，24h 一文不值。

**skill 随时效（相对 persistence，早期计算）**：
+1h **+0.758** → +6h **+0.879（峰值）** → +24h +0.693 → +48h +0.485。
**论文只测 1 小时，错过了自己方法的最佳工作点。**

## 4.5 超参搜索（重要）

**发现：之前所有对比都不公平。**

```
              lr=3e-4   lr=1e-3   lr=3e-3
  3D U-Net     0.0739    0.0716    0.1128     ← 1e-3 最好
  3D CNN       0.0947    0.1298    0.0924     ← 1e-3 最差
  ConvLSTM     0.0954    0.0551    0.0484     ← 3e-3 最好
```

我们统一用的 1e-3 **恰好偏袒 U-Net、坑了 3D CNN**（差 40%）。

**ConvLSTM 之前宽度写死在模型文件里**（756,545 参数 vs U-Net 的 590 万，差 8 倍），
命令行改不了。**已修复**——三个模型现在共用 `--base-channels` / `--depth`。

**调参后每个架构的最佳配置（6 epoch 短训练，只用于排序）**：

```
ConvLSTM   lr 3e-3  base 48   1,687,777 参数   val MAE 0.0393  ①
3D U-Net   lr 1e-3  base 48  13,313,425 参数   val MAE 0.0613  ②
3D CNN     lr 3e-3  base 32   3,913,857 参数   val MAE 0.0924  ③
```

**3D CNN 对容量完全不响应**（98 万 → 880 万参数，误差不变）——
瓶颈是结构（缺 skip connection），不是规模。

⚠️ **这些数字不能写进论文**（短训练），需要用最佳配置重训。**未做。**

## 4.6 浮标独立验证（最新，最有价值）

13 个 NDBC 站，测试年份，**434,662 配对小时**。

**大浪时（浮标 >4 m）的误差分解**：

```
lead    ERA5−浮标    模型−ERA5    继承占比    主导项
  1h     −0.455       −0.017        96%      训练数据
  6h     −0.401       −0.036        92%      训练数据
 24h     −0.453       −0.838        35%      模型      ← 交叉
 48h     −0.453       −2.130        18%      模型
```

**左列近乎常数**（ERA5 固有偏差，与时效无关）；**中列增长 125 倍**。

**结论**：
- **≤6h**：96% 的误差来自训练数据，改架构最多回收 4%
- **≥24h**：模型贡献 65–82%，物理约束在这里才有意义

**附带发现**：长时效时模型**小浪高估**（+0.25 m @48h）、**大浪低估**（−2.13 m），
两头往中间挤 —— MSE 目标的直接后果。

⚠️ **重要修正**：早期我把极端段低估全归因于 MSE 均值回归。
**在 1h 时效上这是错的**——96% 是从 ERA5 继承的。

**局限（写论文必须写）**：
1. >4 m 只有约 550 小时，样本少
2. 浮标是点、ERA5 是 56×46 km 平均，有代表性误差
3. ERA5 低估高海况是已知的；我们的贡献是**量化它在 ML 误差里的占比**

---

# 5. 文件结构

```
src/
  config.py            域、路径、划分年份            ← 被 import，不直接跑
  dataset.py           滑窗、按年划分、masked loss    ← 被 import
  evaluate.py          分层指标                      ← 被 import
  models/              unet3d/cnn3d/convlstm/persistence  ← 被 import

  download_era5.py     ERA5 分块下载（已完成）
  preprocess.py        netCDF → 9.8 GB memmap（已完成）
  train.py             训练
  predict.py           测试集推理
  compare_models.py    自动发现 checkpoint + 出对比表   ← 常用
  build_caches.py      生成各时效预测缓存
  buoys.py             NDBC 下载 + 匹配（已完成）
  validate_buoys.py    浮标三方对比
  bias_decomposition.py 误差来源分解                  ← 核心结果
  hparam_search.py     超参搜索（已完成）
  baseline_table.py    persistence/climatology 参考
  ablation_masking.py  陆地掩膜对指标的影响（已完成）
  cds_queue.py         CDS 队列清理
  smoke_test.py        模型形状/吞吐自测
  pipeline_smoke_test.py 全流程自测（合成数据，几分钟）

notebooks/01_results.ipynb   35 cell，8 图，6 表，零报错
TEACHING_GUIDE.md            820 行中文讲义（gitignore，本地）
_archive/extreme_analysis/   台风/极端事件分析（有 README 说明如何恢复）
_teaching_subset/            538 MB 四年数据子集（gitignore）
outputs/                     checkpoints / figures / tables / logs
data/                        raw + processed（gitignore，约 20 GB）
```

**已训练的 checkpoint（9 个）**：
```
h1  : unet3d, convlstm, cnn3d, unet3d_paper_swh
h6  : unet3d
h24 : unet3d, convlstm, cnn3d
h48 : unet3d
```

---

# 6. 踩过的坑（务必避免重蹈）

| 坑 | 后果 | 已加的保护 |
|---|---|---|
| **不带参数跑 `train.py`** | 默认值命中已训模型文件名，中断后留下 1 轮的权重。**丢过一个训了 4.5 小时的 U-Net** | 检测到同名 checkpoint 会拒绝，需 `--tag` 或 `--force` |
| **`predict.py` 覆盖整张指标表** | 只评部分模型会删掉其他行，**notebook 崩过 3 次** | 改成合并写入 |
| **冒烟测试写真实 outputs** | 清理时把真实结果表删掉，**发生过 2 次** | `WAVE_OUTPUT_DIR` 重定向到沙盒 |
| **notebook 硬取 `.iloc[0]`** | 缺任何数据就在文档中间崩 | 4 处全部改成跳过 + 提示 |
| **归一化不匹配** | 用不同年份的缓存跑旧 checkpoint，静默输出错误米数 | checkpoint 记录 target_stats，不匹配就拒绝 |
| **CDS 单请求上限 121,000** | 整年请求被拒 | 拆成两月块 |
| **CDS 并发队列上限 ~6** | kill 进程后孤儿作业占满配额 | `cds_queue.py --purge` |
| **Windows DataLoader** | spawn 会序列化 9.8 GB memmap | `ERA5Cache.__getstate__` 只传路径 |
| **搜索 checkpoint 误提交** | 223 MB 进了仓库 | 已 gitignore |

---

# 7. 未完成的工作（按优先级）

## 高优先级

**① 用最佳超参重训三个模型**（约 4–6 小时）
现在的对比表用的是统一 lr=1e-3，已知不公平。搜索结果在
`outputs/hparam_search/results.csv`。重训后才有能写进论文的数字。

**② 目录清理**（用户已同意，未做）
把 `ablation_masking.py`、`hparam_search.py`、`baseline_table.py` 移到
`_archive/`，`src/` 加个 README 说明哪些要跑、哪些是被 import 的。

## 中优先级

**③ perfect prognosis 实验**
把未来真实风场也喂进输入，隔离"波浪物理没学好"和"没法预知天气"两部分。
`dataset.py` 改窗口即可，约半天 + 几小时训练。

**④ 输入通道消融**
去掉风场通道，量化"风场信息值多少"。预期 1h 影响小、24h 影响大。

## 低优先级 / 未来

- 近岸降尺度（USACE WIS、Duck FRF 数据）
- 卫星高度计验证（ESA Sea State CCI；注意 ERA5 已同化，不完全独立）
- 和数值模式（NOAA GFS-Wave）对比 —— **期刊会要求**
- MP2 变量（用户目前不要）
- **PINO 架构设计** —— 最终目标

---

# 8. 论文规划

**三张表已就绪**：

| 表 | 内容 | 文件 |
|---|---|---|
| 1 | ERA5 测试集，三架构 × WMO 海况等级 | `outputs/tables/comparison_h1.csv` |
| 2 | 浮标独立验证 × 海况 | `outputs/tables/buoy_validation_h1.csv` |
| 3 | **误差来源分解 × 时效 × 海况** | `outputs/tables/bias_decomposition.csv` |

**第 3 张是独有的**，没有对标。

**期刊查证过的数据**（2026年8月）：

```
Ocean Engineering    中科院 工程技术 1区 TOP，IF 6.31，审稿官方均值 28.6 周
Coastal Engineering  中科院 工程技术 2区，IF 5.51，接受率 43%
```

注意两者显示的是不同版本分区表，**投稿前要在同一版上核对**。

**会议**（已查证）：
- WAVES Workshop：2027年12月6–10日，横滨（**最对口**）
- AGU Fall Meeting 2026：摘要 2026年8月5日截止，**已过**
- ICCE 2026：2026年5月，**已结束**

**这个领域会议只投摘要，不算正式发表，和期刊投稿不冲突。**

**定位建议**（我的观点，用户未最终拍板）：
不在"预测什么"上竞争，在"怎么评估"上竞争。三个独有贡献都是方法论：
误差来源分解、时效维度、超参公平性。

---

# 9. 教学任务（进行中）

用户要给一个高中生（ABC，中文水平不确定）讲这份代码。

- **讲义**：`TEACHING_GUIDE.md`，820 行，24 条预判问答，**中文，仅本地**
- **邮件草稿**：`C:\Users\Minghao\Desktop\email_to_student.md`（英文）
- **教学数据**：`_teaching_subset/` 538 MB（2003/2004/2005/2021 四年）
  ⚠️ 用这个重建缓存后，仓库自带的 checkpoint 会被归一化校验拒绝——这是保护不是故障

讲义结构：课前检查 → 看结果 → 跑起来 → 看代码 → **预判问答** → 动手调参 → 故障排查。

---

# 10. 我在对话中做过的修正（避免新对话重复错误）

1. **"比论文好 6.7 倍"** → 错。论文只在台风条件下测试，我们的是全记录。
   同等条件（TC 分层 + 论文通道）应该是 **0.104 vs 0.140，只好 1.35 倍**。
2. **"3D U-Net 打不过 persistence"** → 错，那是只看前 3 个 epoch。收敛后 skill +0.758。
3. **"极端段低估全是 MSE 均值回归"** → 1h 时效上错，96% 是从 ERA5 继承的。
4. **"Coastal Engineering 比 Ocean Engineering 更受认可"** → 错，后者是 1区 TOP。
5. **"应该加 MP2 凑表格数量"** → 收回，会让工作更像原论文。
6. **"每个模型都低估大浪"** → 3D CNN 的 bias 是正的（但 MAE 最差）。
7. **总结里写的 "+86%"** → 实际是 **+83%**（用完整精度算）。

---

# 11. 用户的工作习惯

- 中文交流，但教学材料和代码注释用英文
- 不喜欢冗长回复，多次要求"简洁"
- **会自己跑命令**——所以任何有破坏性的默认行为都要加保护
- 关心 token 消耗，长任务希望后台跑、只在结束/出错时通知
- 对图表质量有要求（嫌 matplotlib 默认的"普通"）
- 会质疑结论的合理性（"好 10 倍是不是不对"），**这些质疑通常是对的**
