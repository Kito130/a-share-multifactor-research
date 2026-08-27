# A 股多因子研究：技术面试深挖指南

这份文档服务于作者本人复习。面试时应把项目描述为一个横截面研究与回测工程，
而不是单只股票价格预测器。所有数字以仓库中的公开汇总证据为准；证券级输入、
订单和完整审计产物没有随仓库发布。

## 一分钟介绍

我研究了价值、12-1 动量和 60 日低波三个月频横截面信号，依次完成点时股票池、
因子诊断、组合模拟、验证和一次冻结后的最终 OOS。组合在下一交易日开盘交易，
显式记录成本、停牌、涨跌停、失败订单、现金和公司行动。公开仓库只提供合成数据
演示及有限的 OOS 汇总，因此重点是研究纪律、交易约束和可审计的软件实现，而不是
把历史收益当成实盘承诺。

## 研究问题

核心问题是：简单三因子等权组合，在点时股票池、下一开盘成交、交易成本和公司行动
约束下，能否产生风险调整后的横截面超额收益？研究期为 2016--2019，验证期为
2020--2021，最终 OOS 为 2022--2025。验证不重新选择参数；冻结后 OOS 只运行一次。

## 从入口到结果的数据流

```text
scripts/build_p1_panels.py
  -> src/a_share_p1/build.py
  -> daily/month-end normalized panels
scripts/build_p2_single_factor_research.py
  -> src/a_share_p2/research.py
  -> factor panel, Rank IC, quintile diagnostics
scripts/build_p3_backtest.py
  -> src/a_share_p3/build.py
  -> composite signals, targets, orders, portfolio path
scripts/build_p4_validation.py
  -> src/a_share_p4/build.py
  -> validation report and frozen configuration
scripts/build_p5_oos.py
  -> src/a_share_p5/build.py
  -> one authorized final OOS
scripts/build_p6.py
  -> src/a_share_p6/build.py
  -> post-OOS robustness diagnostics
```

配置集中在 `configs/paths.yaml`、`configs/p2_factor_research.yaml`、
`configs/p3_backtest.yaml`、`configs/p4_validation.yaml`、`configs/p5_oos.yaml` 和
`configs/p6_robustness.yaml`。`src/a_share_common/` 只放路径、配置、哈希和原子 I/O，
不放因子或回测含义。

## 信号与数学

对股票 `i`、信号月 `t`：

```text
B/M proxy(i,t)  = 1 / PB(i,t), PB > 0
MOM(i,t)         = P_adj(i,t-21) / P_adj(i,t-252) - 1
LOWVOL(i,t)      = - sample_std(last 60 daily returns)
```

12-1 动量跳过最近约一个月，即用 `t-252` 到 `t-21` 的价格。这样减少短期反转、
月末微观结构和最近一个月信息混入中期动量的影响，也保证因子定义是过去信息的函数。
三个因子每月先做 1%/99% 截面缩尾，再用截面样本标准差标准化：

```text
z_i = (x_i - mean(x)) / sample_std(x)
composite_i = (z_BM + z_MOM + z_LOWVOL) / 3
```

股票池还要求上市满 120 个交易日、20 日流动性足够、PB 为正、因子齐全、无历史
ST 标记、代码区间和上市参考有效。`1/PB` 是 B/M 代理，不等于独立重建的点时
账面权益。

## 执行与会计

月末收盘形成信号，安排在下一个市场交易日开盘成交，卖出先于买入。组合选择最高
的 100 个复合分数，目标等权且单票目标权重上限为 2%。模拟记录现金、失败买卖单、
停牌、开盘涨跌停、佣金/滑点、历史印花税、陈旧估值，以及 `600270.SH` 换股为
`601598.SH` 的人工公司行动。没有声称已经建模排队、容量或完整市场冲击。

常用指标：

```text
Sharpe = mean(daily_return) / sample_std(daily_return) * sqrt(252)
drawdown_t = NAV_t / max(NAV_0..NAV_t) - 1
information_ratio = mean(strategy_return - benchmark_return)
                   / sample_std(excess_return) * sqrt(252)
```

换手是每期交易金额相对组合规模的比例，文档中的 OOS 数字使用双边累计口径。成本
必须同时和初始资金、换手及单边 bps 说明，不能孤立解读。

## 避免未来函数

信号只读取信号月可见的价格、PB、上市、名称、行业和代码历史；下一月收益只作为
评价标签，不参与当月排名。`scripts/build_p2_single_factor_research.py` 的研究
面板在 `2016-01-01..2019-12-31` 内构造，验证和 OOS 在后续入口中隔离。冻结配置、
输入哈希和运行清单在 P4/P5 预检中核对，避免修改输入后悄悄复跑。

## OOS 结果与失败发现

基准情景为单边 10 bps、初始资金 100,000,000 元：累计收益 35.8547%，年化收益
8.2948%，相对中证全指年化差 +8.5163 个百分点，Sharpe 0.5434，最大回撤
-19.8052%，双边换手 18.4729，总交易成本约 4,499,738 元（初始资金约 4.50%）。
这些是排除的许可数据运行后的汇总证据，不是实盘记录。

动量 OOS 平均 Rank IC 为 -0.02325、年化 ICIR 为 -0.5167。不能看到这个结果后
删除动量，因为那会把最终 OOS 变成模型选择集，破坏冻结测试的解释。三只退市持仓
缺少可审计终止事件，结果沿用最后可得复权收盘价；这是重要限制而不是应隐藏的异常。

## 最容易被质疑的五个问题

### 1. 为什么不是时间序列预测？

每个月在同一截面给股票排序，标签是下一月横截面收益；问题是相对选择而非预测
某一只股票的绝对价格。组合结果仍需通过执行和成本模拟才能评价。

### 2. 为什么动量为负还保留？

因为它是冻结前预先指定的因子。OOS 后删因子相当于用测试集做模型选择；正确做法
是披露失败，并把改进留给新的研究协议。

### 3. 为什么不用收盘价直接成交？

收盘价形成信号后不能假设同时成交。使用下一交易日开盘，并明确卖出优先，减少
不可实现成交假设。

### 4. 4.50% 成本是否说明策略不可用？

它说明高换手和资金规模对结果很重要。成本数字必须和初始资金、双边换手及 bps
场景一起看；项目没有据此宣称容量或实盘可行性。

### 5. 合成 demo 能证明什么？

只能验证因子诊断代码的输入输出契约：180 行、6 个月、3 个因子和 18 行 IC。它
不能复现许可数据的正式 OOS，也不能支持真实证券结论。

## 三段现场代码

1. `src/a_share_p2/research.py::_monthly_ic`：按信号月分组，在有下一月标签且
   达到最小横截面样本后计算 Spearman Rank IC。
2. `src/a_share_p3/build.py::_build_composite_signals`：读取冻结因子列，生成复合
   分数、排序和目标持仓，不读取未来收益。
3. `src/a_share_p3/build.py::_simulate_scenario`：按交易日处理订单、现金、持仓、
   成交失败和成本，形成 NAV 路径。

辅助边界代码可看 `src/a_share_common/paths.py::assert_project_relative`、
`src/a_share_common/hashing.py::file_sha256` 和
`src/a_share_common/io.py::atomic_write_json`。

## 最小现场演示

在仓库根目录执行：

```powershell
python scripts/generate_demo_data.py
python scripts/run_public_demo.py
python -m pytest tests/test_common_infrastructure.py tests/test_public_demo.py -q
```

预期摘要为 `SYNTHETIC_SOFTWARE_DEMO`、3 个因子、180 行面板、6 个信号月份和
18 行 top-minus-bottom 结果。干净公开克隆没有许可数据，因此正式 P1--P6 测试会跳过。

## 未实现及原因

- 没有公开证券级行情、订单、持仓或供应商响应：受数据许可和隐私边界限制；
- 没有容量、排队和完整市场冲击模型：现有数据不足以支持可信估计；
- 没有重新运行或调参正式 OOS：保护冻结测试的有效性；
- 没有把 post-OOS 稳健性变成新的 OOS：它只能解释已观察结果，不能制造新证据。

## 复习时应记住的边界

能够回算公式、指出时间边界、解释负面结果和承认数据限制，比背诵收益数字更重要。
面试中不要把 `results/p5_oos/oos_performance.csv` 的汇总描述成实时业绩，也不要把
`data/demo_synthetic/monthly_factor_panel.csv` 描述成真实市场数据。
