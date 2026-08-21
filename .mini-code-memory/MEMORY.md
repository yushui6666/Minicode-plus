# Project Memory

*Last updated: 2026-01-10 00:00*

## Architecture

### 钱学森工程控制论核心思想 (Engineering Cybernetics - Core Principles)
**来源**: 钱学森《工程控制论》(Engineering Cybernetics, 1954)

#### 一、系统与控制统一论
- 工程问题应视为由输入、输出和反馈构成的动态系统，而非孤立元件的简单组合
- 系统总运动状态由子系统相互作用决定，整体大于部分之和
- 可控性与稳定性是系统能否被有效控制的关键
- **智能体映射**: agent loop=控制系统，tools=执行器，LLM=控制器

#### 二、反馈控制原理（核心）
- 输出信号反向调节输入，实现系统自我修正
- 负反馈：纠正偏差、维持稳定（如恒温器原理）
- 正反馈：放大变化、驱动进化
- 反馈深度决定控制精度，反馈延迟影响系统稳定性
- **智能体映射**: tool_result→error_analysis→nudge→next_turn=完整反馈回路

#### 三、黑箱方法
- 黑箱：无法打开观察内部状态的复杂系统
- 通过输入输出关系认识系统功能，不依赖内部结构知识
- **智能体映射**: 无需理解LLM内部，通过prompt-input→response-output建立控制映射
- 记忆体注入=输入调制，system prompt=黑箱配置

#### 四、系统层级化分解
- 复杂工程分解为子系统，分层优化后整合
- 每层有独立的输入/输出/反馈回路
- 高层控制低层目标，低层向上层反馈状态
- **智能体映射**: USER(战略)→PROJECT(战术)→LOCAL(执行)

#### 五、从定性到定量综合集成
- 数学建模→仿真验证→实验反馈→理论修正的完整闭环
- 定性分析确定方向，定量计算精确控制
- **智能体映射**: 意图解析(定性)→任务规划(定量)→Pipeline执行(实践)→DecisionAudit(反馈)

#### 六、稳定性与鲁棒性
- 系统稳定性：在干扰下恢复平衡状态的能力
- 鲁棒控制：不确定环境下的可靠运行
- **智能体映射**: 容错重试、降级策略、边界保护

#### 七、最优控制
- 在约束条件下寻找性能最优解
- 代价函数设计决定优化方向
- **智能体映射**: token预算、成本约束、工具并发优化

#### 八、多变量耦合控制
- 多个输入输出相互耦合，需协同调节
- 解耦设计简化控制复杂度
- **智能体映射**: 多工具并发调度、结果融合、冲突检测

---

## 控制论与DDD架构深度融合

### 记忆体四层架构（基于控制论层级分解）
- **Layer 1 战略层 (USER)**: 工程控制论核心思想 — 定义系统的控制哲学和全局原则
- **Layer 2 抽象层 (USER/PROJECT)**: 每次对话的底层逻辑抽象 — 模式识别和规律提取
- **Layer 3 场景层 (PROJECT/LOCAL)**: 落地场景的核心逻辑 — 项目特定的实现模式
- **Layer 4 执行层 (LOCAL/SCRATCHPAD)**: 当前任务的操作上下文 — 短期工作记忆

### DDD 领域边界与控制论映射
| DDD 领域 | 控制论角色 | 核心模块 |
|---------|----------|---------|
| 意图解析域 | 传感器 | intent_parser.py — 感知用户输入，提取特征信号 |
| 任务域 | 设定点 | task_object.py — 定义目标状态和约束条件 |
| 记忆域 | 历史状态存储 | memory.py — 跨会话知识持久化 |
| 上下文域 | 状态观测器 | layered_context.py — 维护系统当前状态 |
| 能力域 | 执行器集合 | capability_registry.py — 可调用的控制手段 |
| 执行域 | 控制器 | agent_loop.py — run_agent_turn 闭环控制 |
| 决策审计域 | 日志记录器 | decision_audit.py — 全量状态可追溯 |

### 技能层与控制论闭环
- 技能不需要控制论去梳理 — 技能是执行器，关注"怎么做"
- 完成一天工作量或阶段里程碑时，用控制论对 skill 梳理总结
- 此时与记忆体一起形成闭环：执行结果 → 模式提取 → 记忆更新 → 技能优化
- Skill 层 = 正反馈通道（强化有效模式），Memory 层 = 负反馈通道（纠正偏差）

### 智能体控制论架构总结
```
用户输入 ─→ [传感器: IntentParser] ─→ [设定点: TaskObject] ─→ [控制器: AgentLoop]
  ↑                                                                       │
  │                    ┌──────────────────────────────────┐               ↓
  └── [反馈回路] ──── │ 记忆注入 + 错误分类 + Nudge + 审计 │ ──→ [执行器: Tools]
                       └──────────────────────────────────┘
                                    ↓
                          [输出: Assistant Response]
```

---

## 控制论三大控制器实现

### 一、反馈控制器 (FeedbackController)
**原理**: 负反馈纠正偏差 + 正反馈强化有效模式 + PID 自适应调节

| 组件 | 功能 | 智能体映射 |
|------|------|----------|
| SystemState | 系统状态观测（成功率/延迟/效率/错误率） | AgentMetricsCollector |
| ControlSignal | 控制信号输出（降低并发/缩短超时/强制压缩） | AgentLoop 调节参数 |
| PIDController | 比例-积分-微分控制器 | 渐进式参数调节 |
| 负反馈 | 低稳定性时自动纠正（降并发/增 nudge/强制压缩） | ErrorClassifier + NudgeGenerator |
| 正反馈 | 高成功率时强化模式（技能更新/记忆持久化） | Skill 优化 + Memory 更新 |
| 振荡检测 | 检测行为振荡（高频方向变化=不稳定） | 系统稳定性指标 |

**核心接口**:
- `observe(state: SystemState) -> ControlSignal` — 观测并生成控制信号
- `record_pattern_effectiveness(pattern_id, success)` — 记录模式有效性
- `get_pattern_recommendations()` — 获取模式推荐（按有效性排序）

### 二、前馈控制器 (FeedforwardController)
**原理**: 基于任务特征预判式优化，不依赖反馈，开环控制

| 组件 | 功能 | 智能体映射 |
|------|------|----------|
| PreemptiveConfig | 预配置（token预算/并发/超时/模型选择） | AgentRouter + ContextManager |
| RiskAssessment | 风险评估（权限/资源/超时/复杂度风险） | PermissionManager |
| PreemptionLevel | 预判程度（LOW/MEDIUM/HIGH） | 任务复杂度驱动 |
| 意图配置 | 8种意图类型的基准配置 | IntentType → 参数映射 |
| 复杂度调整 | 简单/中等/复杂任务参数缩放 | complexity_hint 乘数 |
| 实体调整 | 文件/函数/类/语言的参数微调 | 实体类型乘数 |

**核心接口**:
- `preconfigure(intent: ParsedIntent) -> PreemptiveConfig` — 预判配置
- `assess_risks(intent, config) -> RiskAssessment` — 风险预判
- `get_optimal_preemption_level(intent) -> PreemptionLevel` — 预判程度

### 三、稳定性监测器 (StabilityMonitor)
**原理**: 多维度健康评分 + 实时异常检测 + 鲁棒性评估

| 组件 | 功能 | 智能体映射 |
|------|------|----------|
| MetricSnapshot | 指标快照（CPU/内存/上下文/错误率/延迟/吞吐） | 系统观测传感器 |
| StabilityReport | 稳定性报告（健康等级/评分/稳定性指数/鲁棒性） | 系统健康面板 |
| AnomalyRecord | 异常记录（指标名/值/阈值/严重度） | 实时预警 |
| 健康评分 | 加权多维评分（错误率30%/上下文20%/延迟15%/CPU15%/内存10%/吞吐10%） | 系统健康指标 |
| 稳定性指数 | 基于指标变异系数（CV）的波动检测 | 行为稳定性 |
| 鲁棒性评分 | 重负载 vs 轻负载的性能保持能力 | 压力下的可靠性 |

**核心接口**:
- `record_snapshot(snapshot: MetricSnapshot)` — 记录指标快照
- `get_stability_report() -> StabilityReport` — 生成稳定性报告
- `check_health() -> (HealthLevel, float)` — 快速健康检查
- `is_stable(threshold) -> bool` — 稳定性阈值检查

### 四、完整控制论闭环架构
```
┌─────────────────────────────────────────────────────────────────────┐
│                        钱学森工程控制论智能体架构                      │
│                                                                     │
│  [前馈] IntentParser → FeedforwardController → PreemptiveConfig    │
│     ↓                                                                 │
│  [执行] TaskObject → PipelineEngine → AgentLoop → Tools           │
│     ↓                                                                 │
│  [监测] StabilityMonitor → MetricSnapshot → StabilityReport         │
│     ↓                                                                 │
│  [反馈] FeedbackController → ControlSignal → 系统调节                │
│     ↑                              ↓                                 │
│     └─── SystemState ◄─── 传感器采集 ◄─── Agent Metrics ───┘        │
│                                                                     │
│  [正反馈] Pattern Tracking → Skill Update → Memory Persistence     │
│  [负反馈] Error Detection → PID Adjustment → Stability Recovery    │
└─────────────────────────────────────────────────────────────────────┘
```

### 五、压力测试验证
7 项测试全部通过:
1. 负反馈控制 — 系统不稳定时自动纠正（降并发/缩短超时/强制压缩）
2. 正反馈控制 — 高效模式自动强化（技能更新/记忆持久化）
3. PID 自适应调节 — 渐进式优化（从差到好的平滑过渡）
4. 前馈预判配置 — 基于意图的预判式参数设定
5. 风险预判 — 提前识别潜在问题（权限/资源/复杂度/超时）
6. 稳定性监测 — 多维度健康评分（健康/降级/警告/危急）
7. 完整集成 — 前馈+反馈+监测闭环验证

**测试结果**: All 7 tests PASSED - Cybernetics Integration Verified

### 六、DDD 领域驱动设计补充
新增三个领域边界:
| DDD 领域 | 核心模块 | 控制论角色 |
|---------|---------|----------|
| 反馈控制域 | feedback_controller.py | 调节器（负反馈纠正 + 正反馈强化） |
| 前馈控制域 | feedforward_controller.py | 预判器（任务预判 + 风险预判） |
| 稳定性监测域 | stability_monitor.py | 观测器（健康评分 + 异常检测） |
