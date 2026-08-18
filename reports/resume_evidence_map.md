# 简历证据映射

> 以下表述均可由项目内文件复核；P6 结果必须注明为事后稳健性分析。

| 可用简历表述 | 核心证据 | 限制/面试口径 |
|---|---|---|
| 构建 A 股三因子月频选股研究，覆盖研究、验证与一次性最终 OOS | `configs/frozen_config.yaml`; `reports/oos/p5_run_manifest.json` | 不声称无偏全市场生产策略 |
| 最终 OOS 2022 至 2025，10 bps 场景年化收益 8.29%、最大回撤 -19.81% | `results/p5_oos/oos_performance.csv`; `reports/oos/p5_result_report.md` | 同时披露退市陈旧估值和 PB 修订政策 |
| 实现停牌、涨跌停、历史印花税及换股吸收合并的事件驱动回测 | `configs/trading_costs.yaml`; `data/manual/corporate_actions.csv`; `reports/backtest/p3_audit_report.md` | 不声称覆盖所有公司行动 |
| 通过 P1 至 P6 分阶段审计、配置冻结与 SHA-256 数据血缘保证复现 | `reports/robustness/p6_run_manifest.json`; `results/experiment_registry.csv` | P6 是 post-OOS，不用于调参 |
| 完成微盘剔除、窗口、行业/规模中性、成本与退市回收敏感性 | `reports/robustness/p6_robustness_report.md`; `results/p6_robustness/experiment_metrics.csv` | 这些不是新的 OOS |

## 数字核对

- 冻结配置：`14ab015fbce37a2236772f69d157c4b9ade05fcb92c0469301de553a22486d68`
- P5 年化收益：8.294818%
- P5 累计收益：35.854726%
- P5 最大回撤：-19.805169%
- P5 输出哈希项数：45
- P6 稳健性重新运行变体数：5
