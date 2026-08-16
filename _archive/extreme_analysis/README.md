# 归档：极端海况分析

**这些文件不是废弃代码，是暂时移出主目录的。**

2026-08-05 移到这里，原因是要给高中生讲解代码，主目录只保留常规 SWH 预报的
部分，避免混淆。**后续写论文时这部分是核心贡献之一，需要移回去。**

---

## 里面是什么

### `src/`

| 文件 | 作用 |
|---|---|
| `hurdat2.py` | 下载并解析 NHC 大西洋飓风路径库（HURDAT2），计算到 Hatteras 的距离 |
| `eda_hurdat2.py` | 风暴目录统计分析，产出 `01_` `02_` 两张图 |
| `extremes.py` | **从波高场本身检测极端事件**，并用 HURDAT2 标注 TC/ET 来源，产出 `03_` 图和 `extreme_events.csv` |

两个 `.txt` 文件是从主目录代码里**移除的片段**，保留原样以便恢复：

| 文件 | 原本在哪 |
|---|---|
| `_regime_strata_removed_from_predict.py.txt` | `src/predict.py`，按 TC/ET 风暴类型分层评估 |
| `_check_extremes_removed_from_smoke_test.py.txt` | `src/pipeline_smoke_test.py` 的 STEP 5，极端事件检测的自测 |

### `outputs/`

| 文件 | 内容 |
|---|---|
| `01_storm_tracks_domain.png` | 1979–2024 年经过 Hatteras 500 km 内的 148 场风暴路径 |
| `02_storm_statistics.png` | 逐年次数、季节分布、强度分布 |
| `03_extreme_events.png` | 76 场极端波浪事件的月份分布、强度-持续时间、逐年统计 |
| `extreme_events.csv` | **极端事件目录**，每场事件带 TC/ET 标签和可审计的匹配信息 |
| `hurdat2_hatteras_events.csv` | HURDAT2 风暴筛选结果 |
| `hurdat2_closest_30.csv` | 距离最近的 30 场风暴 |
| `test_metrics_regime_swh_h*.csv` | 按 TC/ET/background 分层的模型评估指标（4 个时效） |

### `data/`

| 文件 | 说明 |
|---|---|
| `hurdat2_atlantic.txt` | HURDAT2 原始数据，1851–2025，2004 场风暴（7.1 MB） |
| `hurdat2_tracks.parquet` | 解析后的 55,605 个轨迹点 |

---

## 这部分得出的主要结论

写论文时这几条是要用的：

1. **论文的方法在这个区域失效**。用台风路径切数据，只覆盖 3.0% 的小时记录，
   丢掉 97% 的训练数据；46 年里只有 5 场 3 级以上飓风在 Hatteras 附近。

2. **冬季完全是空白**。HURDAT2 在 12/2/3/4 月零事件——而这正是 Outer Banks
   东北风暴（nor'easter）的季节。

3. **改用波高场定义极端事件后**（P99 阈值、持续 ≥12 h），2000–2025 共 76 场：
   - 温带风暴 60 场（**78.9%**），共 1,523 小时
   - 热带气旋 16 场（21.1%），共 525 小时
   - **12–3 月的 58 场事件全部是温带风暴，热带气旋零场**
   - 两类平均强度几乎相同（ET 9.15 m vs TC 9.14 m），差别只在频次
   - 26 年最大的一场浪 **12.00 m 出现在 2014-03-27**，是场东北风暴，
     HURDAT2 里查不到

4. **模型在两类风暴上表现不同**。+1 h 时 ET 上 skill +0.858、TC 上 +0.704；
   到 +48 h 时反转成 TC +0.467、ET +0.348。

5. **极端段的系统性坍缩**。P99+ 分层里，bias/MAE 从 +1 h 的 66% 涨到
   +48 h 的 **100%**——每一个极端点都低估，一次都没高估过。

---

## 怎么恢复

```bash
cd "C:\Users\Minghao\Desktop\Ocean Wave"

# 1. 代码
move _archive\extreme_analysis\src\extremes.py     src\
move _archive\extreme_analysis\src\hurdat2.py      src\
move _archive\extreme_analysis\src\eda_hurdat2.py  src\

# 2. 数据
move _archive\extreme_analysis\data\hurdat2_atlantic.txt   data\raw\
move _archive\extreme_analysis\data\hurdat2_tracks.parquet data\interim\

# 3. 结果（可选，重跑脚本也能重新生成）
move _archive\extreme_analysis\outputs\*.png outputs\figures\
move _archive\extreme_analysis\outputs\*.csv outputs\tables\
```

然后把两个 `.txt` 里的代码片段贴回 `predict.py` 和 `pipeline_smoke_test.py`。

`predict.py` 里还需要恢复调用点，原本在写完百分位分层表之后：

```python
    regimes = regime_strata(t_index, cache, valid, lead)
    if regimes:
        print("\n" + "#" * 88)
        print("# Storm-regime stratified")
        df_r = evaluate(predictions, truth, valid, strata=regimes)
        print(format_report(df_r, target, lead))
        df_r.to_csv(TABLES / f"test_metrics_regime_{target}_h{lead}.csv",
                    index=False)
```

`config.py` 里的 HURDAT2 相关常量（`HURDAT2_URL`、`TROPICAL_STATUS`、
`SAFFIR_SIMPSON`、`classify_intensity` 等）**没有移走**，恢复时不用改。
