# EOS 与热力学通道审计

## 决策

**当前数据不得从 `press` 转换为 `eint`。**

本项目后续的 paper-adapted reduced reproduction 必须继续使用：

```yaml
thermal_channel: press
paper_adaptation: true
eos_conversion: disabled_unverified_gamma
```

原因不是转换公式未知，而是当前 111 个原始文件、处理后 HDF5 和仓库配置中都没有
保存与这次 GRMHD 运行相匹配的 EOS 类型和绝热指数 `Gamma`。在理想气体/adiabatic
EOS 假设成立时，通常可写 `eint = P/(Gamma-1)`；但没有 run-specific `Gamma` 时，
代入任意常见值都会制造一个无法追溯的新物理量；尤其不得猜 `Gamma=4/3` 或 `5/3`。

## 审计范围与可复核证据

审计日期为 2026-07-17。检查对象包括：

- `data_raw/` 所指的全部 111 个 ATHDF/HDF5 原始快照；
- 规范处理文件 `data_proc/grmhd_regrid_inner_r200_64.h5`；
- 当前仓库的配置、脚本、README 与元数据；
- 本机 `/home/curl/athena-public-version` 中可见的公开 Athena 源码和示例输入；
- 论文正文及 Appendix A/C。

规范处理文件信息：

| 项目 | 值 |
| --- | --- |
| 实际目标 | `/home/curl/datasets/grmhd-no/grmhd_regrid_inner_r200_64_n111.h5` |
| SHA-256 | `cd3966bed1f617f49d775151cff87ae92eb2432a560bb82ae9355605e7aea70a` |
| shape | `(111, 8, 64, 64, 64)` |
| 通道 | `[Bcc1,Bcc2,Bcc3,rho,press,vel1,vel2,vel3]` |
| 时间范围 | `0` 到约 `1100.000692` |
| 最小 rho | `3.7619026898028096e-06` |
| 最小 press | `1.9604057044375622e-08` |

111 个原始文件均可打开，且根属性的 schema 一致。可见属性只有：

```text
Coordinates, DatasetNames, MaxLevel, MeshBlockSize, NumCycles,
NumMeshBlocks, NumVariables, RootGridSize, RootGridX1, RootGridX2,
RootGridX3, Time, VariableNames
```

数据对象上也没有额外 EOS 属性。所有快照共同显示：

| 字段 | 一致值 |
| --- | --- |
| `Coordinates` | `kerr-schild` |
| `DatasetNames` | `prim`, `B` |
| `VariableNames` | `rho,press,vel1,vel2,vel3,Bcc1,Bcc2,Bcc3` |
| `MaxLevel` | `3` |
| `MeshBlockSize` | `[22,4,16]` |
| `NumMeshBlocks` | `2020` |
| `RootGridSize` | `[88,32,16]` |
| `RootGridX1` | `[1.1,1200,1.08273038]` |

未找到 `gamma`、`gamma_adi`、`adiabatic_index`、`eos` 或等价字段。处理后 HDF5
同样没有这些信息；其元数据只足以说明这是球坐标 Kerr--Schild、对数 r 网格上的
nearest-cell-centre AMR 重网格，而且该重网格不是守恒或 `div B` 保持变换。

## 找到但明确排除的证据

本机 Athena 公共源码树中存在示例文件：

```text
/home/curl/athena-public-version/inputs/mhd_gr/athinput.fm_torus
```

该示例写有 `gamma = 1.4444444444444444`，但它的自旋、根网格和外边界分别为
`a=0.5`、`64^3` 和约 `rmax=35`，与当前数据的文件命名和
`RootGridSize=[88,32,16]`、`rmax=1200` 明显不匹配。它只能证明 Athena 输入文件
可能在哪里记录 gamma，不能证明当前运行用了 `13/9`，因此被排除。

论文 Appendix A 对 Newtonian MHD 写明理想气体 `gamma=5/3`，但 GRMHD 小节没有
给 gamma。Newtonian 设置不是当前 GRMHD 数据的 EOS provenance，不能把 `5/3`
移植到这里。

Athena 源码可确认 adiabatic ideal-gas 实现通常使用
`P=(Gamma-1)eint` 这一关系；源码仍不能回答本次运行选择了哪个 `Gamma`、是否启用
了完全相同的 EOS 编译/运行设置。因此公式可知不等于转换获准。

## `press` 与论文 `eint` 的语义差异

当前原始变量清单把热通道明确命名为 `press`，且它位于 primitive 数据集。论文的
表述并不完全一致：Appendix A 把 GRMHD 输出写成 `rho,P,v,B`，Appendix C 又把训练
通道写成 `dens,eint,vel,bcc`。在没有论文数据生成代码或 EOS metadata 的情况下，
不能假设 Appendix C 的 `eint` 只是 `P` 的别名。

这会影响的不只是通道标签，还包括：

- 正值对数变换中的取值分布与 epsilon；
- 论文径向基线使用的 `U=dens+eint`；
- envelope 和 bounds 的训练统计；
- 温度代理 `T=P/rho` 的定义；
- 与论文逐通道相对 L2 数字的可比性。

因此当前可做的是把同一套数值方法适配到 `rho/press`，并在每个配置和结果文件中
标记 thermal-channel adaptation；不能只把数组名称从 `press` 改成 `eint`。

## 单位与坐标信息的限制

当前 HDF5 属性没有记录密度、压力、长度或时间的单位制，也没有匹配运行输入来证明
采用了哪组 code units。论文 GRMHD 描述中的单位选择不能自动归属于这批数据。

此外，当前 `vel1/2/3` 和 `Bcc1/2/3` 保持原始球坐标 Kerr--Schild 的坐标分量语义；
重网格过程没有把它们变成论文所写的 Cartesian `x/y/z` 分量。EOS 审计不会顺便做
基矢变换，后续模型元数据必须保留这一点。

## 若未来找回匹配输入，转换必须满足的门槛

只有同时满足以下条件，才允许重新评估 `press -> eint`：

1. 找到与当前 111 个快照同一次运行的输入文件、编译配置或输出 metadata；
2. 由文件直接确认 EOS 类型和 `Gamma`，并能用网格、spin、时间、文件前缀等字段
   与当前数据建立唯一匹配；
3. 确认 `press` 是该 EOS 下的 gas pressure，而不是总压、温度代理或其他变量；
4. 由代码和一个独立数值检查确认转换公式及 code units；
5. 审计记录包含来源文件 hash、字段位置、解析值和适用范围。

即使门槛满足，也不得覆盖当前规范 HDF5。应按项目约定在外部数据目录创建独立文件：

```text
/home/curl/datasets/grmhd-no/grmhd_regrid_inner_r200_64_n111_eint.h5
```

新文件应至少保存：

- 原文件 SHA-256 和新文件 SHA-256；
- `source_thermal_channel=press`、`thermal_channel=eint`；
- EOS 名称、精确 `Gamma`、转换公式和证据文件 hash；
- 完整通道顺序、坐标系统、单位说明和生成代码 commit；
- 转换前后 positivity、finite-value、min/max/quantile 统计；
- 数值 round-trip 检查 `P -> eint -> P` 的最大绝对/相对误差。

若这些条件中的任何一项缺失，决策保持为 `eos_conversion: disabled_unverified_gamma`。

## 对后续复现实验的约束

- 不修改、不覆盖当前 111-snapshot HDF5；
- 不重命名 `press` 以伪装成 `eint`；
- normalizer、radial baseline、bounds 和 loss 的统计全部从 reduced 训练切分重新拟合；
- 配置、checkpoint metadata、summary 和图表都必须显示 `thermal_channel: press`；
- 与论文 `eint` 指标的任何对照必须标记为不可直接比较；
- Round 1--3 的既有结果保持冻结，其 `press` 语义不追溯修改。

本轮没有生成转换数据，也没有启动训练。

## Stage J 再审计（2026-08-01）

Stage J 对所有 111 个 raw header、项目 metadata、以及本机 Athena 公共源码树重新扫描，
没有发现能够改变上述结论的新 provenance。机器可读结果保存为
`outputs/paper_reduced100/stage_j/eos_audit.json`。

公共 Athena 源码树当前 commit 是
`9c266692b9423743d8e23509b3ab266a232a92d2`，工作树干净。其中
`src/eos/adiabatic_mhd_gr.cpp` 的 SHA-256 为
`cb248d8d3a33c288abd198d887cac5d43c0716f7bcb6c5e97f455749d58efcc3`；构造函数从
`hydro/gamma` 读取数值，GRMHD 焓项使用 `gamma/(gamma-1)`。这只验证了该份公共代码
的一种实现约定。无法证明 raw 文件由这个 commit、这个 EOS 模块或任何特定 gamma
生成。

再次找到的 `athinput.fm_torus` SHA-256 为
`672e82615a53057c75eede91a0b4be344d97ab881da3ca61349110c12c518d68`。它的
`a=0.5`、`64^3` root grid 和约 35 的外边界与 raw header 的
`[88,32,16]`、`rmax=1200` 不匹配，继续归类为 unrelated sample，而不是 matching
run config。论文的 GRMHD 正文/Appendix A 写压力 `P`，Appendix C 写 `eint`，但没有
给出当前数据可用的 GRMHD Gamma；Newtonian 小节的 `5/3` 也不是本运行证据。

因此 Stage J 的最终状态保持：

```yaml
thermal_channel: press
eint_status: BLOCKED_UNVERIFIED_EOS
gamma: null
eos: null
conversion_authorized: false
```

本次再审计没有计算或写入 `press/(gamma-1)`，没有修改任何 HDF5。
