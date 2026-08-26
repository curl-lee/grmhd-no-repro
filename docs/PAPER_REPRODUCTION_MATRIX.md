# 论文复现矩阵

## 审计结论

本文件审计的是论文方法与本仓库之间的可追溯关系，不是对既有 Round 1--3
结果的重新解释。审计基准如下：

- 论文：[arXiv:2512.01576v1](https://arxiv.org/abs/2512.01576) 与
  [NeurIPS ML4PS 2025 论文 PDF](https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_56.pdf)；
- 上游 `neuraloperator`：固定提交
  `86a8bc7812a31b42c4f7895693cf4ac11521c066`；
- 当前发布分支：`main`；
- 审计日期：2026-07-17；
- 当前数据：111 个时刻、8 通道、球坐标 Kerr--Schild 重网格数据；热通道是
  `press`，不是经 EOS 验证的 `eint`。

结论是：当前仓库已经有一批可直接复用的上游软件组件，也实现了若干受论文启发的
变换、先验和损失；但它尚不是论文协议的忠实复现。主要阻塞项是数据规模和几何不一致、
`eint` 缺少可验证转换依据，以及上游 3D LocalNO 不支持论文需要的 3D DISCO 局部积分层。
因此后续实验必须命名为 **paper-adapted reduced reproduction**，不能命名为 exact
paper reproduction。

## 状态定义

| 状态 | 含义 |
| --- | --- |
| `EXACT` | 在所声明的范围内，公式或上游软件路径逐项一致；不自动代表整套论文实验一致。 |
| `ADAPTED` | 有对应实现，但因数据、几何、定义或缩减预算而发生实质适配。 |
| `BLOCKED` | 缺少当前不能合理补出的数据、物理元数据或上游能力。 |
| `EXTENSION` | 本仓库自己的研究扩展，不属于论文复现协议。 |
| `NOT_IMPLEMENTED` | 论文中有定义，但当前仓库尚无对应实现。 |

## 逐项矩阵

| 论文组件 | 论文位置/公式 | 上游源码位置 | 当前实现 | 状态 | 差异 | 下一步 |
| --- | --- | --- | --- | --- | --- | --- |
| 300 个快照 | 正文 Method：`N_data=300` | 无数据实现 | 当前规范 HDF5 只有 111 个快照 | BLOCKED | 无法从现有数据补出 189 个真实时刻 | 保留全 111 快照的 provenance；协议明确标注 reduced data |
| 使用最后 250 个快照 | Results：300 个快照中的最后 250 个 | 无 | 当前有 `all111`、`late100` 等时间窗 | BLOCKED | 当前总量小于 250，不能照搬窗口 | reduced 主协议固定使用最后 100 个快照，并记录绝对索引与时间 |
| 80/20 训练/验证 | Results：最后 250 个按 80/20 划分 | `Trainer` 接收外部 loader，不定义切分 | `PaperReduced100Protocol` 固定最后 100 个为 train 11..90 / validation 91..110，丢弃 90->91 transition，无 test | ADAPTED | 80/20 顺序语义已实现，但数据是 reduced100 而非论文 last250 | Stage F paired configs、manifest、loader 与 79/19 pair tests 已锁定 |
| Cartesian Kerr--Schild `64^3` | Appendix A，GRMHD 数据描述 | 无 | `64^3` 球坐标 Kerr--Schild 网格，轴顺序为 `(phi,theta,r)` | ADAPTED | 分辨率相同，但坐标拓扑、体元、方向和边界均不同 | 配置和报告强制写出 `spherical_ks_adaptation: true`；不得称 Cartesian |
| 8 个状态通道 | Appendix A：`rho,P,vx,vy,vz,Bx,By,Bz`；Appendix C：`bcc1:3,dens,eint,velx:y:z` | `FNO`/`LocalNO` 支持任意 `in_channels/out_channels` | `[Bcc1,Bcc2,Bcc3,rho,press,vel1,vel2,vel3]` | ADAPTED | 通道数一致，但顺序、坐标基矢和热通道语义不同；论文自身 `P`/`eint` 表述也不一致 | 固定当前通道清单和坐标基；输出元数据声明 `thermal_channel: press` |
| `eint` 热通道 | Appendix C “Data” 与归一化段 | 无 | 数据只有 `press` | BLOCKED | 原始/处理后文件均无 run-specific `Gamma`/EOS，不能验证 `eint=P/(Gamma-1)` | 不转换；详见 `EOS_AND_THERMAL_CHANNEL.md`；仅在找回匹配运行输入后另建 HDF5 |
| 下一状态监督 `u_t -> u_{t+DeltaT}` | 正文 Method | `DataProcessor` 与 `Trainer` 的 next-step/rollout 接口 | `GRMHDNextStepDataset` + `PaperDataProcessor` + paired paper-state configs | ADAPTED | 映射和 tensor space 已锁定；数据热通道/几何仍适配 | Stage F real-batch、manual/upstream Trainer parity、checkpoint/AR tests 已通过 |
| 正值通道变换 | Appendix C：`log10(x+epsilon)`，用于 `dens/eint` | 上游 Gaussian normalizer 不提供该变换 | `GRMHDNormalizer` 对 `rho/press` 做正值对数 | ADAPTED | `press` 代替 `eint`；当前 epsilon 是固定常数，不是训练子集估计 | 按论文规则只从 reduced 训练集估计 epsilon；保留 `press` 适配标志 |
| 有符号磁场变换 | Appendix C：`sgn(x) log10(1+abs(x)/epsilon)` | 无 | `GRMHDNormalizer` 对三个 B 分量采用同形变换 | ADAPTED | 变换公式对应，但 epsilon 当前为固定 `1e-6` | 改为训练集 `max(abs(x))` 量级规则，并保存每通道统计量 |
| 速度线性通道 | Appendix C：velocity uses identity transform | 无 | 三个速度分量保持线性 | EXACT | 仅此通道变换公式一致；坐标分量仍是球坐标基 | 保持公式，元数据记录 `vel1/2/3` 的坐标基语义 |
| epsilon 估计 | Appendix C：正值用 `10^(floor(log10(min x))-2)`；有符号用 `10^(floor(log10(max abs(x)))-2)` | 无 | 当前 epsilon 固定为 B `1e-6`、rho `1e-12`、press `1e-14` | ADAPTED | 未按训练子集估计，可能改变动态范围 | 仅扫描训练切分，保存 epsilon、样本范围和计算版本 |
| median/MAD 稳健缩放 | Appendix C：`m=median(xhat)`，`s=1.4826 median(abs(xhat-m))`，下限 `1e-6` | 上游 normalizer 不是该公式 | `GRMHDNormalizer.fit` 实现 median、1.4826 MAD 与 scale floor | ADAPTED | 公式一致，但输入受固定 epsilon、`press` 和可选旧 radial baseline 影响 | paper-adapted 路径独立拟合，禁止读取旧 Round 统计文件 |
| gamma soft clip | Appendix C：`gamma tanh(z/gamma)`，`gamma=6`；逆变换先截到 `0.99 gamma` | 无 | `GRMHDNormalizer` 同公式、同 `gamma=6`、同逆截断 | EXACT | 公式级一致；不代表其前后处理整体一致 | 添加 reduced 统计 round-trip 与饱和率审计 |
| 8 个径向 shell 通道 | Appendix C “Position features”：索引/Cartesian 半径，边界在 `[1e-1,rmax]` 对数均分，默认 `rmax=10` | 模型仅负责接收额外通道 | `radial_shells_tensor` 对物理球坐标 `r` 做 8 个 one-hot 几何分箱 | ADAPTED | 当前 shell 只沿球坐标 r；论文按三维网格中心的 Euclidean/index radius | reduced 协议保留球坐标 shell，但命名 `spherical_r_shells`，不得写成论文 shell 的 exact 实现 |
| 径向 baseline `B(r)` | 正文 Scaling Priors；Appendix C：对 `U=dens+eint` 回归 `log10(U)` 与 r，使用同一 `B(r)=k r` | 无 | train-only `appendix_literal_press_proxy`：rho+press 无截距 literal fit，再按 train shares 分通道 | ADAPTED | press 代替 eint，球坐标 r；Stage F 代码确认只作 envelope reference，不做 state subtraction | Full envelope 读取；Plain 保持同一 direct normalized state，representation test 锁定 |
| baseline-centered envelope | Appendix C：仅 `dens/eint`，以 `B(r)` 为中心，容差 `Delta=1.5`，权重 `0.05` | 无 | `ResidualEnvelope` 对所有通道的一步 `y-x` 残差拟合训练分位数 | ADAPTED | 当前是数据驱动全通道残差包络，不是论文固定的热/密度径向包络 | 为论文路径单独实现 rho/press-adapted 固定 envelope；旧包络只保留为诊断扩展 |
| 训练物理分位数 bounds | Appendix C：物理域 `q=.001/.999`，再加 5% multiplicative margin 后映射到归一化空间 | 无 | `QuantileBounds.fit` 直接在编码值上取分位数 | ADAPTED | 缺少物理域拟合和 5% margin；映射阶段不同 | 从训练物理状态拟合，保存 raw quantiles、margin 和编码后 bounds |
| 评估时 rho/eint clamp | Appendix C：解码前在归一化域 clamp `dens/eint` | `DataProcessor.postprocess` 可承载该逻辑 | `PaperDataProcessor.decode_prediction` 仅在 evaluation clamp rho/press 并报告命中率 | ADAPTED | press 代替 eint，bounds 来自 reduced train-only physical quantiles | Full/Plain one-step 与 3-step AR 均走同一 evaluation-only clamp；Plain training loss隔离测试通过 |
| 通道加权 L2 | Appendix C：B 权重 1.2，velocity 1.0，dens/eint 1.0 | `LpLoss` 可作绝对平方 parity，但其默认调用是 relative L2 | `PaperComponentFidelityLoss`：每通道空间 MSE、通道和、batch mean，B 权重一次 | ADAPTED | 论文未写 reduction divisor；按 Appendix D “per-voxel MSE”解析并将 eint 适配为 press | Stage E 手算、batch、channel、上游 parity 和 gradient audit 已通过 |
| H1 项 | Appendix C：默认权重 `lambda_H1=0.05`；源文总式存在一次系数记号歧义 | `neuralop.losses.H1Loss` | `PaperH1GradientLoss` 复用 upstream absolute squared H1 并减去 L2，只乘一次 0.05 | ADAPTED | Stage H 证实三轴均按周期 unit-cube `1/N` 差分：相对 unit-index 能量放大 4096；不是球坐标/GRMHD covariant gradient | 论文主权重与 Stage G 结果保持冻结；Stage I alternative H1 均单列为 diagnostic extension |
| Stage I no-H1 诊断 | 不属于论文主路径；用于隔离 spherical-grid H1 adaptation | 仍复用同一 upstream FNO/Trainer | Run A 仅将 selected H1 training contribution 置零，保留 Full 的 base/ROI/bounds/envelope/dissipation、共同初始状态与 pair order；完成 30 epochs/600 steps、strict reload 和 100-step rollout | EXTENSION | 不等于 Plain L2；validation average 0.48282，GT rollout 1/5/10/19 均优于 Stage G Full，step 50/100 norm 更低，但 clipping fraction 仍为 1.0 | Run A 已完成；与 Run B 共同进入部分比较，论文主 Full 不变 |
| Stage I unit-index H1 诊断 | 不属于论文主路径；用于移除 current H1 的 `64^2` spacing amplification | upstream FNO/Trainer；本地冻结 centered periodic stencil | Run B 只把 selected H1 换为 spacing `[1,1,1]`、uniform voxel mean、weight 0.05；其余 Full 项与 paired state/order 不变；完成 30 epochs/600 steps、strict reload 和 100-step rollout | EXTENSION | validation average/global 0.48249/0.38547；H1/base gradient ratio 0.00475，clip scale 0.14299；GT 1/5/10/19 均优于 Full/no-H1，但 Plain 仍在多数形态/边界类最佳 | Run B 部分判定 A；stored-coordinate/volume proxy 尚未执行，Stage I 最终判定待后续单独授权 |
| velocity ROI | Appendix C：解码真实速度，物理 speed top 20%，`kappa=8`，375 epochs 线性 ramp，mask 内相对范数 | 无 | canonical oracle stored-component proxy top 20%，normalized velocity relative L2，逐 sample mask | ADAPTED | 当前分量是 spherical Kerr--Schild stored-component proxy，不是 metric-correct speed | Stage D/Stage E mask、ramp、batch、zero-gradient 和 clamp-overlap audit 已通过 |
| dissipative constraint | Appendix C：训练集全局归一化 L2 范数阈值 `Rmax`，`Rin=1.05Rmax`，`Rout=1.5Rin`，sigmoid `beta=10`，增长权重 `5e-4` | 无 | `PaperDissipativeReference.apply` 与 `PaperCompositeLoss` 直接复用 train-only reference | ADAPTED | normalized global array norm，不是物理耗散率；validation norm shift 不参与重拟合 | Stage E value/zero-subgradient/per-sample batch/gradient audit 已通过 |
| rho/eint 下界与上界罚项 | Appendix C：低界权重 0.05；上界默认关闭 | 无 | Stage D normalized rho/press bounds 由 Full loss 复用；低界 0.05、上界精确零 | ADAPTED | press 代替 eint；validation violation 只报告 | channel isolation、zero-weight、truth-equals-prediction 和 gradient audit 已通过 |
| 3D LocalNO backbone | 正文与 Appendix C：3D LocalNO，8+m 输入、8 输出 | `neuralop.models.LocalNO` | 当前有上游 `LocalNO` 的 3D differential-only proxy | ADAPTED | 类可在 3D 使用 differential kernels，但论文所需局部 DISCO integral 未启用 | 保留为 `LocalNO-diff-only` 基线，不得简称论文 LocalNO |
| 3D DISCO 局部积分层 | 正文 LocalNO；Appendix C | `local_no_block.py` 对非 2D local conv 明确抛出 `NotImplementedError` | 无可运行的 3D DISCO 路径 | BLOCKED | 固定上游提交只实现 2D local DISCO integral | 在不改动固定上游的前提下单独实现/上游化 3D DISCO，或明确使用 FNO proxy |
| FNO 软件复用 | 论文最终 backbone 不是 FNO；仅作为本项目 proxy | `neuralop.models.FNO` | `src/grmhd/models.py` 直接包装固定上游 FNO | EXACT | 这是上游软件来源的 exact 复用，不是论文 backbone 的 exact 复现 | 继续作为可运行 proxy；报告中与论文 LocalNO 分栏 |
| Trainer 软件复用 | 论文给训练协议，不指定该库 Trainer 内部实现 | `neuralop.training.Trainer` | `PaperTrainerAdapter` 调用固定上游 `train_one_batch`；仅本地传 epoch/accumulation/logging | EXACT | `EXACT` 只指核心 upstream batch path 未复制；外层工程控制是本地适配 | Stage F manual/Trainer output、component、gradient、ramp parity 已通过 |
| Adam 软件复用 | 论文 Appendix C 写 Adam | PyTorch optimizer 与上游 Trainer 可组合 | Stage G paired entry 使用 `torch.optim.Adam(lr=1e-3,wd=1e-4)` | EXACT | 只表示 optimizer 算法/超参对应；reduced epoch 预算仍适配 | Full/Plain 各完成 600 次 finite update，并严格重载 optimizer state |
| 论文优化与调度协议 | Appendix C：Adam、`lr=1e-3`、`wd=1e-4`、75 epoch warmup、cosine 至 `1e-6` | Trainer/调度器可配置 | paired config 同时保存 paper 1200/75 与 actual 30/2；Stage G 使用 accumulation 4 和 cosine 至 `1e-6` | ADAPTED | epoch/warmup 因 reduced 预算缩为 30/2 | 两个 30-epoch pilot 已完成；结果仅用于 resource-scaled 判断，不启动 1200 epoch |
| 上游 LpLoss/H1Loss | Appendix C 的 L2/H1 概念 | `neuralop.losses.LpLoss`、`H1Loss` | 已用于上游基线/对齐测试；主自定义 loss 另实现 | EXACT | 只指类实现的原样复用；论文复合损失仍是 ADAPTED | 分别测试上游 loss 和论文复合 loss，报告不要混称 |
| checkpoint/training state | Appendix C 提到早停与训练超参 | 上游 `training_state.save/load_training_state` | Stage G `best_validation_l2/last` 复用 upstream bundle，并保存配对、order、loss 与 provenance metadata | EXACT | 论文不规定文件格式；这里只声明上游 state 复用 | 两模式 model/Adam/scheduler/epoch/ramp/order/prediction/validation parity 均通过 |
| 1200 epochs 与有效 batch 16 | Appendix C：1200 epochs，batch 4，accumulation 4，patience 100 | Trainer 支持配置化训练 | Stage G 实际为 30 epochs、batch 1、accumulation 4、79 microbatches/epoch | ADAPTED | 受单卡与 reduced-data 预算限制；每 epoch 最后一次只累积 3 batches | 两模式各完成 600 optimizer steps；报告始终同时标注 paper 1200 与 actual 30，不直接数值比较 |
| CNN ablation | Appendix D：3D U-Net，base 32，levels 32/64/128/256，residual 3x3x3、GroupNorm、SiLU | 无对应专用模型 | 未实现 | NOT_IMPLEMENTED | 当前仓库没有该论文 CNN | 先写模型形状和参数量测试，再允许训练 |
| Fourier PE ablation | Appendix D：归一化 Cartesian 坐标，`L=4`，24 通道 | FNO/LocalNO 可接收额外通道 | 未实现 | NOT_IMPLEMENTED | 当前只有球坐标 radial shells，没有 Fourier PE | 若实现，必须定义球坐标适配版并与论文 Cartesian 24 通道区分 |
| no-shell ablation | Appendix D：移除位置通道和 radial loss nudge，保留其他项和 `B(r)` | 模型输入通道可配置 | `configs/ablations/no_shell.yaml` 只关闭当前 shell | ADAPTED | 现有配置属于旧协议，且没有同步论文 radial-loss 细节 | 基于新的 paper-adapted 基准生成完整派生配置并做 resolved-config diff |
| no-radial/constraint ablation | Appendix D：移除 baseline、envelope、constraint penalties 和 eval clamp，保留其他项 | 无 | `no_radial_baseline.yaml` 仅关闭当前 baseline | ADAPTED | 现有 ablation 不完整，不能对应论文表中的 No Radial | 新配置一次性关闭论文列出的全部组件，并用测试断言 |
| plain L2 ablation | Appendix D：无加权 normalized MSE，禁用全部辅助项/权重 | `LpLoss.abs(take_root=False)` 可作 parity | 正式 `plain_l2_fno.yaml` + strict `PlainL2Loss`；Stage G 与 Full 共享初始状态及每 epoch 数据顺序 | ADAPTED | loss 解析对应；press/球坐标/reduced preprocessing 与论文数据仍不同 | 30-epoch Plain 已完成并胜出全部冻结 morphology/stability 类别；先诊断 Full 的 H1 adaptation |
| 50/100-step rollout | Appendix D：50 与 100 steps；论文说明长期模拟可能没有 ground truth | Trainer 有通用 autoregressive rollout；当前 eval 脚本也能滚动 | Stage G 从 snapshot 91 完成 19-step GT 和 100-step physical autoregression | ADAPTED | reduced validation 只有 19 个 GT transitions，step 20 后无 GT error | 已输出 step 1/5/10/19 GT 与 step 50/100 finite、positivity、range、morphology和统计；两模式均 finite |
| 归一化 volumetric relative L2 (%) | Results/Table 2：逐通道和平均百分比 | `LpLoss`/评估循环可计算相对范数 | Stage G 输出逐通道、算术平均、global，并分列 canonical oracle、persistence、Full、Plain | ADAPTED | 聚合口径已锁定，但球坐标/reduced/press/FNO proxy/30 epochs 不能与论文数字直接横比 | 保留 normalized 主 checkpoint 指标，并与 physical oracle-aware 诊断分表 |
| 统计观测量 | Appendix A：球平均 rho、`T=P/rho`、质量吸积率积分 | 无 | eval 有若干统计摘要，但没有经过论文公式/metric 一致性验证的完整三项 | NOT_IMPLEMENTED | 需要度规体元、四速度/坐标变换等输入；当前 8 通道未直接提供完整 `u^r` 与度规 | 先做可计算性审计；rho/T 可适配，Mdot 缺物理字段时标为不可识别，禁止伪造 |
| 长期统计性质 | Results：长期统计比 chaotic transient 的逐点误差更重要 | 评估循环可扩展，无专用实现 | Stage G 保存 shell mean/std/quantile、autocorrelation、PSD、total variation、high-k energy 与 artifact flags | ADAPTED | 当前是 spherical reduced proxy 统计，不含论文 coarse/fine 运行时语义 | Full/Plain 已同口径比较；Stage G 判定 Plain 长期形态更稳，暂停消融 |
| 最外层边界统计 | Results：fine level 的外边界向 coarse level 提供信息 | 无 | Stage G 正式输出最外 1--2 spherical shells 的逐步统计与时间变化 | ADAPTED | 论文 Cartesian 外边界与当前 spherical outer shells 不同，且无真实耦合 | Plain outer-shell score 优于 Full；只能作为离线 spherical adaptation 诊断 |
| 外边界耦合 | Results：fine level 在其外边界向 coarse level 提供动态信息 | 无 | 未实现 | NOT_IMPLEMENTED | 当前仅离线 HDF5 surrogate，无运行中的求解器边界接口 | 作为独立耦合项目，不纳入 reduced 单步复现的成功判据 |
| coarse/fine coupling | Appendix E：两个 NO 时刻线性插值；覆盖 coarse-grid inner hydro boundary；磁场经 CT/EMF 替换 | 无 | 未实现 | BLOCKED | 缺少匹配 coarse/fine AthenaK 求解器、边界接口和 CT/EMF 数据 | 取得论文耦合代码或定义可验证的 AthenaK 接口后另开工作流 |
| 零初始化有界残差 | 不属于论文 | 无 | `ZeroInitializedResidual`/bounded residual 路径 | EXTENSION | 本项目为改善自回归稳定性加入 | 保留为扩展对照，表格和文件名加 `extension` |
| hybrid physical target | 不属于论文 | 无 | `HybridTargetStats` 与 `target_mode="hybrid"` | EXTENSION | 论文训练的是下一状态，不是当前 signed-additive/positive-log-ratio hybrid target | 不进入 paper-faithful 主线；可作为后续稳定性对照 |
| decoded/rollout-aware custom loss | 不属于论文 | 上游 Trainer 被项目子类扩展 | `GRMHDRolloutTrainer` 与 decoded/rollout/range 组合 | EXTENSION | 论文复合 loss 定义不同 | 与 paper-adapted state loss 分开配置、分开结论 |
| Fold B | 不属于论文 | 无 | `src/grmhd/fold_b.py` | EXTENSION | 本项目的 leakage-safe 时间折设计 | 仅作为泛化诊断，不替代论文 80/20 reduced split |
| recency weighting | 不属于论文 | 无 | Round 3 采样策略 | EXTENSION | 论文未报告该采样权重 | 只在扩展实验中使用并报告有效样本权重 |
| 自定义 5% persistence gate | 不属于论文 | 无 | 当前 summary/筛选逻辑的 persistence 改善门槛 | EXTENSION | 论文没有该通过/失败定义 | 继续作为项目工程判据，但不得用于声称论文指标复现 |

## 论文内部需要保留的歧义

复现代码不能偷偷替论文作决定，以下差异必须写入 resolved metadata：

1. Appendix A 的 GRMHD 输出写作 `rho,P,v,B`，Appendix C 的 8 通道却写作
   `dens,eint,vel,bcc`。当前证据不能证明二者在论文数据中如何转换。
2. 正文以 `log u approximately -k|x|+b` 描述径向衰减，Appendix C 的操作定义则
   用 `B(r)=k r`，并以 `U=dens+eint` 做无显式截距的离线最小二乘描述。后续实现应
   选定 Appendix C 的可执行定义，同时在报告中保留这一歧义。
3. Appendix C 的 H1 小节在局部定义和总损失式中存在可能重复写入系数的记号歧义。
   默认超参明确给出 `lambda_H1=0.05`，实现应只应用一次，并用单元测试锁定。

## reduced reproduction 的允许表述

在上述阻塞项解除前，允许的结论是：

> 在 111-snapshot 球坐标 Kerr--Schild、`press` 热通道数据上，对论文方法进行
> paper-adapted reduced reproduction，并用固定上游 `neuraloperator` FNO 或
> differential-only LocalNO 作为可运行 proxy。

不允许使用以下表述：

- “复现了论文的 300-snapshot Cartesian GRMHD 数据”；
- “使用了论文的 `eint`”，除非新的独立数据文件有可验证 EOS provenance；
- “复现了 3D DISCO LocalNO”，除非 3D local integral 层真实可运行并通过测试；
- 把本仓库的 hybrid、Fold B、recency 或 persistence gate 当成论文组件。

## 四个复现层级

- **Level A — exact software reuse**：固定上游提交中的 FNO、Trainer、
  `LpLoss`/`H1Loss`、训练状态/checkpoint 路径，以及直接使用的 PyTorch AdamW。
- **Level B — paper-method adapted reproduction**：reduced100、稳健变换、球坐标 shell、
  spherical-r baseline、bounds/clamp、ROI、index-grid H1、dissipation 和 `press`
  thermal adaptation。
- **Level C — blocked exact reproduction**：300/last-250 数据、Cartesian Kerr--Schild
  数据、经确认的 `eint`、3D DISCO LocalNO 和 coarse/fine coupling。
- **Level D — diagnostic extensions**：zero-initialized bounded residual、hybrid target、
  decoded/rollout-aware loss、Fold B、recency weighting 和 custom 5% persistence gate。

状态矩阵是逐组件判断；四个层级是报告口径。某一软件对象为 `EXACT`，不会把依赖它的
整套实验自动提升为 Level A。

## Table 2 论文参考值

以下数值只用于锁定未来聚合与抄录测试，不是当前数据上的目标阈值。论文列顺序为
`Bx, By, Bz, rho, e, vx, vy, vz`，单位均为 validation relative L2 百分比。

| Configuration | Avg | Bx | By | Bz | rho | e | vx | vy | vz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours | 14.02 | 16.71 | 17.01 | 14.73 | 10.98 | 11.25 | 13.47 | 14.04 | 13.94 |
| PE (Fourier) | 13.87 | 16.77 | 17.06 | 14.87 | 9.85 | 10.42 | 13.73 | 14.08 | 14.15 |
| No PE/Radial Shell | 13.93 | 16.84 | 17.15 | 14.89 | 10.03 | 10.42 | 13.87 | 14.18 | 14.06 |
| No radial/constraint | 14.17 | 16.84 | 17.26 | 14.87 | 10.87 | 11.33 | 13.82 | 14.29 | 14.07 |
| Plain L2 | 13.69 | 16.76 | 16.98 | 14.67 | 9.55 | 9.89 | 13.45 | 14.06 | 14.16 |
| CNN backbone | 19.09 | 23.85 | 23.91 | 21.44 | 15.21 | 14.87 | 17.46 | 17.80 | 18.17 |

论文自身已经显示 Plain L2 的平均 validation L2 低于 Ours；因此未来不能把“Full
proxy 单步 L2 必须优于 Plain L2”设成唯一通过标准。形态、长期统计和边界稳定性必须
与 L2 分开报告。

## 关键源码定位

- 上游 FNO：`external/neuraloperator/neuralop/models/fno.py:25`；
- 上游 LocalNO：`external/neuraloperator/neuralop/models/local_no.py:24`；
- 3D local DISCO guard：
  `external/neuraloperator/neuralop/layers/local_no_block.py:215`；
- 上游 Trainer：`external/neuraloperator/neuralop/training/trainer.py:27`；
- 上游 `LpLoss`/`H1Loss`：
  `external/neuraloperator/neuralop/losses/data_losses.py:21` 与 `:215`；
- 当前 normalizer：`src/grmhd/normalizer.py`；
- 当前 shell、bounds/envelope、复合 loss：`src/grmhd/shells.py`、
  `src/grmhd/priors.py`、`src/grmhd/losses.py`；
- 当前物理态 AR processor：`src/grmhd/data_processor.py`；
- 当前 diagnostic extensions：`src/grmhd/residual.py`、`src/grmhd/hybrid.py`、
  `src/grmhd/trainers.py`、`src/grmhd/fold_b.py`。

Stage F 已完成 paired Full/Plain 工程 smoke；状态为
`passed_with_h1_warning`。Full 的 H1/value 与 gradient ratio 偏大且所有 update clipping，
但两模式均 finite、checkpoint 可严格重载、one-step/3-step 闭环通过。该结果允许进入两个
受控 30-epoch Stage G pilot，并要求继续监控 H1 与 clipping；不构成科学优劣结论。

Stage G 随后在 RTX 5070 上顺序完成两组 pilot。两模式共享完全相同的初始 tensor state
和 30 个 epoch 的 pair order，各完成 600 optimizer steps、严格 best/last reload、
19-step GT 及 100-step physical rollout。Full 的算术平均 one-step L2 略低，但 Plain
的 global L2、三磁场通道和全部冻结 morphology/boundary 类别更好；两者单步均不如
persistence。Full 的 H1/base 比例持续偏高，两模式 clipping fraction 均为 100%。
因此当前判定是 **B：暂停论文消融，先诊断 index-grid H1 adaptation**，且不修改论文
主 H1 权重。

Stage H 已完成该 post-hoc 诊断，没有重新训练。Pinned-source 与 toy parity 证实 H0
使用三轴周期 centered difference 和 `1/N` spacing；相对相同 stencil/reduction 的
unit-index 能量固定放大 `4096=64^2`。内两层 shell 占 H1 的 51.0%（voxel 25%），
`r/theta` 合计占 99.7%；paper-weighted H1 占 post-hoc total gradient norm 平均
94.3%，norm-1 clip 平均 scale 为 0.00608。判定为 **A：index-grid H1 adaptation
mismatch strongly supported**。这一判定不覆盖 Stage G 比较，也不改变论文主
`H1 weight=0.05`。

Stage I 已顺序完成三个单独授权的 EXTENSION：Run A no-H1、Run B unit-index H1 和
Run C stored-coordinate/volume H1。三者各完成 30 epochs/600 finite updates、
best/last strict reload、19-step GT 和 100-step physical rollout。Run C 使用实际
phi/theta/physical-r centers、open theta/r boundaries 和归一化 spherical-coordinate
volume proxy；它不是 covariant GRMHD 或 Kerr--Schild proper-volume H1。Run C best
validation average/global 为 0.44427/0.36254，mean H1/base gradient ratio 0.09180。
三个 extension 均相对 Full 改善五类冻结 morphology/boundary 指标，但相对 Plain
分别只改善 1/1/0 类。最终判定为 **D：No extension rescues Full**。Stage G Full
仍是 paper-adapted 主路径；Run A/B/C 仅解释 adaptation mismatch，不启动后续消融。

## Stage J fidelity blocker resolution

Stage J 对固定上游、本地 refs、111 个 raw snapshots、处理数据、EOS 候选、Kerr--Schild
component semantics、重网格和 coarse/fine 接口做了只读/接口级审计。它不改变上述 Stage
G/I 结果。

| Stage J gate | status | matrix consequence |
| --- | --- | --- |
| 3D DISCO | `REQUIRES_NEW_IMPLEMENTATION` | 固定上游真实 DISCO 只支持 2D；当前 FNO 和 LocalNO diff-only 均非 paper backbone |
| Cartesian KS | `BLOCKED_COMPONENT_BASIS` | raw header 缺 spin、Cartesian convention、速度/B basis 与匹配源码；不得重命名分量 |
| eint/EOS | `BLOCKED_UNVERIFIED_EOS` | 继续使用 `press`；Gamma/EOS/units 均未验证，conversion unauthorized |
| 300 snapshots | `NOT_REPRODUCIBLE_FROM_CURRENT_PROVENANCE` | 本地只有连续 111 个 snapshot，无 author manifest 或可靠 rerun inputs |
| coupling contract | `CONTRACT_READY` | tensor/time/ghost-zone toy contract 可测试；paper API 仍 ambiguous，且缺 paired coarse/fine/CT data |

当前 exact reproduction 仍为 Level C blocked。唯一下一主线是 **B：data acquisition
first**；需先获得 author dataset/run/operator/interface provenance，再决定 geometry
conversion、3D operator implementation 与 coupled validation。Stage J 的完整依赖和证据见
`PAPER_STAGE_J_DECISION.md` 与 `PAPER_STAGE_J_BLOCKER_MATRIX.md`。

## Stage K adapted differential LocalNO execution

Stage K 不改变 Stage G--J 的历史结论，也不解除 exact 3D DISCO、Cartesian KS、verified
`eint`、300 snapshots 或 coarse/fine coupling 阻塞。它只执行 Stage J 已证明可运行的公开
3D differential LocalNO 路径，并将完整 standalone surrogate 流程归类为
`adapted_method_reproduction`。

| Stage K component | implementation/result | status | boundary |
| --- | --- | --- | --- |
| 3D LocalNO constructor | `n_dim=3`, spectral 4, differential Conv3d 4, DISCO/Conv2d 0, 358,296 parameters | ADAPTED | spectral+differential public proxy，不是 paper 3D DISCO |
| data/preprocessing | reduced100 79/19 pairs, spherical KS stored components, `press`, canonical stats SHA256 `164571...4001`, eight input-only shells | ADAPTED | 111 snapshots；无 Cartesian/eint 转换 |
| Plain L2 | 与 Stage G Plain 完全相同的 normalized eight-channel equal-weight MSE | EXACT | `EXACT` 仅指本仓库冻结 loss contract parity |
| pairing | seed 42、Stage G pair order、相同 optimizer/scheduler/batch/accumulation/eval；LocalNO 独立 initial state | ADAPTED | 架构不同，明确不伪造与 FNO 相同 tensor state |
| real 64-cube preflight | RTX 5070 forward/Plain-L2/backward finite；all trainable gradients finite；backward 前后 state hash 相同 | EXACT | 工程 gate；无 optimizer/checkpoint |
| two-epoch smoke | 158 microbatches、40 updates、best/last strict reload、3-step physical rollout passed | EXACT | 仅工程结果，不参与 persistence 排名 |
| formal pilot | 30 epochs、2370 microbatches、600 updates、best epoch 22、无 nonfinite；clipping fraction 1.0 | ADAPTED | 论文预算 1200 epochs；不直接横比 paper Table 2 |
| one-step validation | LocalNO average/global 0.624225/0.540968；persistence 0.189088/0.235074 | ADAPTED | 同一 reduced split；LocalNO 未超过 persistence |
| physical rollout | GT 19 steps、no-GT 100 steps finite/positive；transform counters 100；无 Rout exceedance | ADAPTED | finite 不代表物理正确；step>19 无 GT error |
| morphology/boundary | 复用 Stage G 阈值；Bcc3 在 selected no-GT 25/50/75/100 均有 collapse flag，vel3 在 75/100 有 flag | ADAPTED | Bcc3/vel3 有 preprocessing oracle floor，但不事后改阈值 |
| final decision | `C — TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE` | COMPLETED | 不启动 Full-adapted LocalNO、调参、延长训练或论文消融 |

Stage K 证明的是 adapted engineering workflow 能从冻结 HDF5 运行到 strict checkpoint、
oracle-aware validation 和 physical autoregression；它没有复现 paper 的 3D DISCO LocalNO
或论文数值，也没有证明该输出可作为真实 GRMHD surrogate。完整证据见
`PAPER_STAGE_K_LOCALNO_SOURCE_AUDIT.md`、`PAPER_STAGE_K_LOCALNO_PLAIN.md`、
`PAPER_STAGE_K_ROLLOUT.md` 与 `PAPER_STAGE_K_DECISION.md`。

## Stage L post-hoc collapse attribution

Stage L 不改变任何训练结果，只解释 Stage K 的冻结 collapse flags。Phase 1 严格复现
detector；Phase 2 在相同 checkpoint/rollout 和冻结 `< 0.5` retention contract 下完成
raw→oracle→model 分解。

| Stage L evidence | Bcc3 | vel3 | matrix consequence |
| --- | --- | --- | --- |
| canonical preprocessing: global variance | severe 5/5 | severe 5/5 | 两通道都存在明显 preprocessing floor |
| canonical preprocessing: shell/radial | severe 5/5 | severe 5/5 | 主要结构损失集中于 inner/middle radial regions |
| canonical preprocessing: demeaned combined high-k | severe 5/5 | severe 5/5 | raw→oracle 已移除大部分 stored-index high-k energy |
| LocalNO added: global variance | severe 4/5 | severe 4/5 | 模型继续压缩全局方差 |
| LocalNO added: shell/radial | severe 4/5 | severe 4/5 | 模型继续损害径向/壳层结构 |
| LocalNO added: demeaned combined high-k | severe 0/5 | severe 0/5 | 不是 uniform low-pass；高频被保留/各向异性重分配 |
| channel decision | `C. MIXED_PREPROCESSING_AND_MODEL` | `C. MIXED_PREPROCESSING_AND_MODEL` | overall `3. MIXED_OVERALL` |

`136` 个 near-zero-denominator ratio 被明确记为 `null` 并排除，不以零或任意大值替代；
所有核心分类证据仍有足够 defined steps。FFT 仅是 `(phi,theta,r)` stored-index-space
diagnostic，不是 Kerr--Schild invariant spectrum，最大 Parseval relative error 为
`6.0833e-16`。step 19 之后没有使用 oracle/GT error。

Stage K 最终判定保持 `C — TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE`。Stage L 不授权
训练、改 detector threshold、重 fit preprocessing、修改 checkpoint 或生成新的 long
rollout。详见 `PAPER_STAGE_L_COLLAPSE_DETECTOR_AUDIT.md`、
`PAPER_STAGE_L_PREPROCESSING_FLOOR.md`、`PAPER_STAGE_L_VARIANCE_SPECTRUM.md`、
`PAPER_STAGE_L_FNO_VS_LOCALNO.md` 与 `PAPER_STAGE_L_DECISION.md`。

## Stage M transform floor and oracle transport

Stage M 仅做 post-hoc transform/evaluation audit，不创建新 rollout 或训练状态。它把
Stage L 的 raw→oracle floor 与 oracle→model degradation 进一步拆开，但不修改 Stage L
分类。

| Stage M component | result | status | boundary |
| --- | --- | --- | --- |
| real transform trace | nonlinear → robust z → tanh softclip → inverse clamp → inverse softclip/scaling/nonlinearity | EXACT | 精确指当前代码路径；不是 paper 原作者实现的外部验证 |
| Bcc2 floor source | `E. MULTIPLE_COMPONENTS` | DIAGNOSTIC | clamp-only 与 no-softclip 均恢复 4/4；后者包含 clamp-domain removal，不能解释为独立物理机制 |
| Bcc3 floor source | `A. FORWARD_SOFTCLIP_DOMINATED` | DIAGNOSTIC | encoded 出现 exact ±6；no-clamp atanh nonfinite，no-softclip 恢复 4/4 |
| vel3 floor source | `A. FORWARD_SOFTCLIP_DOMINATED` | DIAGNOSTIC | 与 Bcc3 相同的 saturation/recovery 结构 |
| LocalNO shell/radial transport | 三主通道均未通过 joint persistence-relative gate | ADAPTED | stored-coordinate structural diagnostic，不是 invariant flux transport |
| legacy detector | 保留 `std(pred)/std(raw snapshot 91)<0.05` | FROZEN | floor-limited channel 不得只凭该 flag 判 model failure |
| candidate gate | Gate 0 engineering、Gate 1 floor、Gate 2 model degradation、Gate 3 transport | PROPOSAL | `counterfactual_gate_replay_only`；不追溯改判历史结果 |
| Stage M LocalNO decision | `4. MIXED_TRANSPORT_AND_PREPROCESSING_FAILURE` | COMPLETED | preprocessing floor 与 model-added transport/state failure 均存在 |
| candidate decision | `I. CANDIDATE_GATE_READY_FOR_FUTURE_RUNS` | PROPOSAL_READY | 阈值未用 Stage K/validation model outcomes 调整；尚非 validated physical metric |

## Stage N transform prototypes and operator response

Stage N is isolated and no-training. Prototype statistics use train snapshots
11–90 only, validation is a one-time read after parameter/selection freeze, and
the epoch-22 LocalNO always uses canonical preprocessing.

| Stage N component | result | status | boundary |
| --- | --- | --- | --- |
| Bcc2 prototype | P2B no-softclip selected from train only; validation L2 `1.48e-7` | `A. PROTOTYPE_READY` | P2A validation is not allowed to change train selection |
| Bcc3 prototype | P1 no-softclip; validation L2 `2.03e-7` | `A. PROTOTYPE_READY` | oracle round-trip only |
| vel3 prototype | P1 no-softclip; validation L2 `2.11e-8` | `A. PROTOTYPE_READY` | oracle round-trip only |
| P3 combined | Bcc2/Bcc3/vel3 no-softclip, other channels canonical | `1. READY_FOR_SHORT_PILOT` | not used by a model; no Stage N training authorization |
| R8 fixed-point | variance ratios global/shell/radial median `0.882/0.904/0.853`; delta shrink `0.390` | LOW_VARIANCE_ATTRACTOR supported | exactly two applications |
| shell/radial response | median direct shell/radial skill `-4.17/-6.82` vs persistence | SHELL_RADIAL_TRANSPORT_BIAS supported | stored-coordinate diagnostic, not invariant transport |
| frequency/channel response | 112/112 two-epsilon gain warnings | inconclusive/inconclusive | candidate signatures cannot satisfy reproducibility condition |
| LocalNO operator decision | `4. MIXED_OPERATOR_RESPONSE_FAILURE` | COMPLETED | does not overwrite Stage M classification |
| oracle-conditioned gate | unified legacy + Gates 0–3, version `stage_m_v1` | `I. REPORTING_INTERFACE_READY` | reporting-only; cannot block training |

Stage K/L/M historical decisions and all exact-reproduction blockers remain
unchanged. A future two-epoch P3 smoke requires separate authorization.

## Stage O P3 differential LocalNO pilot

Stage O is the separately authorized execution of the frozen Stage N P3
prototype with the Stage K differential LocalNO. P3 statistics remain isolated,
train-only, and non-canonical; the model remains a public 3D spectral plus
differential proxy with no DISCO integral layer.

| Stage O component | result | status | boundary |
| --- | --- | --- | --- |
| P3 parity | all train/validation snapshots; 344 values; max scaled difference `0.0` | EXACT_LOCAL | identity with frozen Stage N artifact, not author preprocessing |
| target-channel floor | validation Bcc2/Bcc3/vel3 `1.48e-7/2.03e-7/2.11e-8` median | PROTOTYPE_RECOVERY | independently fitted train snapshots 11--90 only |
| architecture | 358,296 parameters; four spectral + four differential modules; DISCO/Conv2d 0 | ADAPTED | not paper 3D DISCO LocalNO |
| GPU preflight | real `(1,16,64,64,64)` forward/loss/backward finite on RTX 5070 | EXACT_LOCAL | engineering gate only; state unchanged, no update |
| two-epoch smoke | 158 microbatches, 40 updates, strict reload, 3-step physical rollout passed | EXACT_LOCAL | complete-data engineering smoke |
| formal pilot | 30 epochs, 2,370 microbatches, 600 updates, best epoch 23, no nonfinite | ADAPTED | resource-scaled, Plain L2, clipping fraction 1.0 |
| one-step validation | P3 normalized average/global `1.04711/0.34434`; persistence `0.18027/0.16827` | FAILED_SKILL | model is `5.80871x` persistence by average |
| oracle-aware validation | P3 floor average `0.07114`; model-to-P3-oracle average `1.963e20` | FAILED_RANGE | Bcc1 dominates physical error; all predictions exceed frozen Rout |
| GT/no-GT rollout | 19/100 steps finite and positive; exact counters | ENGINEERING_PASS | decoded Bcc2/Bcc3 expand to float32 extrema; finite is not stable |
| `stage_m_v1` | Gate 0 range/Rout fails; Gate 2 Bcc2/Bcc3 fail; Gate 3 all target channels fail | FAILED_STRUCTURE | Gate 1 uses P3 floor; no no-GT future target is fabricated |
| decision | `C. P3_PILOT_COMPLETE_BUT_ROLLOUT_UNSTABLE` | COMPLETED | no new model, loss, ablation, tuning, or longer run authorized |

P3 demonstrably removes the old Bcc2/Bcc3/vel3 transform floor, but the model
does not convert it into stable behavior. Legacy collapse flags decrease while
ripple/stripe flags and a more serious decoded-range explosion emerge; therefore
legacy detector improvement is not a stability improvement. Stage K--N decisions
and every exact-reproduction blocker remain unchanged.

## Stage P extreme-range closed-loop attribution

Stage P is a frozen-checkpoint, no-training audit. It does not modify the P3
prototype, LocalNO parameters, Stage O artifacts, or historical decisions.

| Stage P component | result | status | boundary |
| --- | --- | --- | --- |
| provenance/replay | Stage O best epoch 23 and last epoch 30 model-only reload; selected physical states bitwise equal | EXACT_LOCAL | no optimizer/scheduler construction or checkpoint write |
| train envelope | snapshots 11--90 only; validation indices used `[]` | EXACT_LOCAL | diagnostic quantiles, not fitted model parameters |
| first failure | Bcc2/Bcc3/vel3 normalized OOD and Rout condition at step 1 | FAILED_FIRST_STEP | OOD and Rout are coincident, not ordered |
| target inverse clamp | Bcc2/Bcc3/vel3 policies `no_softclip`; occupancy `0` | EXACT_LOCAL | signed-log inverse can still amplify tails |
| teacher forced | 19/19 predictions above frozen Rout | FAILED_ONE_STEP | no feedback, so recursion is not the sole cause |
| normalized direct | state norm grows `19.138x` through step 19 | FAILED_RECURRENT_DIAGNOSTIC | no decode/encode; not deployable |
| roundtrip only | oracle/one-step target states nearly fixed; no repeated-H accumulation | WEAK_FEEDBACK | at most two H applications, no model call |
| projection/reset | Bcc2/Bcc3/vel3 meet frozen primary-driver rule; no unique dominant channel | DIAGNOSTIC_DRIVER_SUPPORT | counterfactual projection/teacher reset is not performance |
| local gain | 120/120 finite, 0/120 pass two-epsilon consistency | INCONCLUSIVE | no Jacobian spectral-radius claim |
| mechanism decision | `6. MIXED_CLOSED_LOOP_FAILURE` | COMPLETED | M1/M2/M5 supported; M3/M4 weakly supported |

The attribution explains why P3 floor recovery did not produce a stable Stage O
rollout: normalized overshoot exists on the first prediction, Bcc2/Bcc3 inverse
tails turn it into extreme physical magnitude, and cross-channel interventions
show multiple drivers. It does not identify a single dominant fix and does not
authorize training. Stage O remains
`C. P3_PILOT_COMPLETE_BUT_ROLLOUT_UNSTABLE`; all exact 3D DISCO, Cartesian KS,
verified `eint`, 300-snapshot, and coarse/fine coupling blockers remain.

Stage K 仍为 `C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE`，Stage L 仍为
`3. MIXED_OVERALL`。Stage M 不把 diagnostic counterfactual 称为 improved preprocessing，
也不解除 exact 3D DISCO、Cartesian KS、verified `eint`、300 snapshots 或 coupling blockers。

## Stage Q persistence-anchored output contract

Stage Q is a frozen-checkpoint, no-training, one-factor contract audit. It uses
the Stage O epoch-23 P3 LocalNO without changing parameters or transforms.

| Stage Q component | result | status | boundary |
| --- | --- | --- | --- |
| contract | `y_alpha = z + alpha(F(z,s)-z)`; scalar grid 0/0.125/0.25/0.5/1 | EXACT_LOCAL | no channel/state/time-dependent alpha or clipping |
| identity/safety | Q1--Q8 pass; parameter/buffer hash unchanged | PASSED | no backward, optimizer, scheduler, or checkpoint write |
| calibration split | 79 train transitions, validation indices read before selection `[]` | EXACT_LOCAL | validation cannot tune alpha |
| overshoot | q-OOD reduction 92.42%/79.79%/70.19% for alpha 0.125/0.25/0.5 | IMPROVED_DIAGNOSTIC | lower OOD is not stability or physical correctness |
| frozen range gate | 79/79 Rout failures for every alpha, including persistence | FAILED_READINESS | no threshold or transform adjustment |
| persistence bound | candidate/persistence normalized average 1.077/1.297/1.893 | FAILED_READINESS | maximum allowed ratio 1.05 |
| structure | Gate-2 reduction below 50%; shell and radial skill worse than persistence | FAILED_READINESS | nontrivial residual alone is insufficient |
| validation | controls only: persistence average 0.180266, direct 1.047111 | CONFIRMATORY | no candidate selection or tuning |
| local response | direct median directional gain 5.211, maximum 257.979 | DIAGNOSTIC | finite differences, not a Jacobian spectral radius |
| rollout | `not_run_no_candidate` | NOT_RUN_BY_RULE | no fabricated candidate or transform counters |
| decision | `D. NO_VALID_ANCHOR_CANDIDATE` | COMPLETED | stop this contract route; no smoke authorization |

Overshoot suppression is real but does not satisfy the frozen stability and
dynamical-skill predicates. Stage O remains
`C. P3_PILOT_COMPLETE_BUT_ROLLOUT_UNSTABLE`; Stage P remains
`6. MIXED_CLOSED_LOOP_FAILURE`; Stage K--N and all exact-reproduction blockers
remain unchanged.

## Stage R normalized-residual P3 LocalNO pilot

Stage R is a newly trained `adapted_residual_contract_model_pilot`, not Stage Q
alpha anchoring. P3, the Stage K architecture/initial state, pair order, data,
optimizer, scheduler, and budget are frozen; only the output/target contract
changes to `delta_z = z_{t+1}-z_t` and `z_hat = z_t+r_theta`.

| Stage R component | result | status | boundary |
| --- | --- | --- | --- |
| contract/equivalence | unscaled 8-channel residual; identity reconstruction; residual/state Plain L2 equal | EXACT_LOCAL | no alpha, scale, output repair, or clipping |
| preflight/smoke | RTX 5070 64-cube pass; 158 microbatches/40 updates; strict reload/3-step pass | ENGINEERING_PASS | failed first launch was a pre-update diagnostic-local bug and was fixed/tested |
| formal training | 30 epochs; 2,370 microbatches; 600 updates; best epoch 9; no nonfinite | ADAPTED | resource-scaled; clipping fraction 0.986667 |
| one-step normalized | average/global `0.224673/0.169252` vs Stage O `1.047111/0.344339` | IMPROVED_VS_DIRECT | still `1.246344x` P3 persistence |
| residual skill | mean cosine/sign `0.1319/0.5242`; residual L2 `1.4138` vs zero baseline `1.0` | INSUFFICIENT | GT cosine reverses after step 1 |
| range/Rout | all validation above Rout; physical explosion by GT step 3 | FAILED_STABILITY | no Stage R physical output clamp |
| Gate 2 / Gate 3 | Bcc2 improves; Bcc3 still fails / all three transport skills negative | MIXED / FAILED | stored-coordinate oracle-conditioned diagnostics |
| 19/100 rollout | finite, positive, exact `100/19/19/100` counters | ENGINEERING_PASS | float32-extreme decoded values and widespread ripple flags are unstable |
| decision | `C. RESIDUAL_PILOT_COMPLETE_BUT_ROLLOUT_UNSTABLE` | COMPLETED | no further training or ablation authorized |

This improves Stage O normalized overshoot but does not demonstrate dynamics
better than persistence. Stage K--Q history and every exact 3D DISCO, Cartesian
KS, verified `eint`, 300-snapshot, 1,200-epoch, and coupled-simulation blocker
remain unchanged.
