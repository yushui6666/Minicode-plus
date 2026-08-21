# MiniCode-plus

## LangGraph — Slice 1-5 (Slice 5 默认)

> `minicode/graph/builder.py:build_model_graph` + `minicode/graph/runtime.py:run_graph_turn` | 默认走图，`MINICODE_USE_GRAPH=0` / `runtime:{"useGraph":false}` 回退旧循环

所有回边收敛到 **策略检查**，Slice 5 新增 `store/on_thinking_delta` 透传、`ModelSwitcher` 兜底重试、回调对齐。

```mermaid
flowchart TD
    HUMAN_START([👤 用户]) -- "用户提问/回答" --> START([开始])
    START --> LC[加载上下文]
    LC --> CP[压缩上下文]
    CP --> SP{策略检查}
    SP -- "终止原因已置 (预算耗尽)" --> FZ[收尾]
    SP -- "仍有剩余步数" --> MD[模型推理]
    MD --> CS{决策分类}
    CS -- "路由=去鉴权 (工具调用)" --> AUTH{鉴权}
    AUTH -- "许可=允许" --> EX[执行工具]
    AUTH -- "许可=拒绝" --> FZ
    EX --> OBS{观测工具}
    OBS -- "终止原因=等待用户 (暂停轮次)" --> HUMAN[👤 人工介入<br/>ask_user / 等待用户回答]
    HUMAN -- "用户已回答 → 新一轮图调用 (同thread_id检查点恢复)" --> FZ
    OBS -- "继续" --> V{校验}
    V -- "请求修复=是 (需证据且修复器存在)" --> RP[修复]
    RP --> SP
    V -- "已校验/无需修复" --> SP
    CS -- "路由=去扩容 (兜底且需扩容)" --> WD{扩容}
    WD -- "无需扩容" --> FZ
    WD -- "已扩容 (奖励步数+1)" --> SP
    CS -- "路由=助手跟进 (进度/重试/守卫/暂停)" --> AF[助手跟进]
    AF --> SP
    CS -- "路由=去收尾 (完成/兜底/阻塞/错误)" --> FZ
    FZ --> END([结束])
    END -- "完成/等待用户 → 回到人" --> HUMAN_END([👤 用户])

    %% 人机边界：虚线表示图外人工
    HUMAN -.->|下一轮恢复| LC
    HUMAN_START -.-> HUMAN
    HUMAN_END -.-> HUMAN_START

    style SP fill:#e8f0fe,stroke:#4285f4
    style CS fill:#fef7e0,stroke:#fbbc04
    style AUTH fill:#fce8e6,stroke:#ea4335
    style V fill:#e6f4ea,stroke:#34a853
    style FZ fill:#f3e8fd,stroke:#8430ce
```

### Human-In-The-Loop

```mermaid
flowchart LR
    U([👤 用户]) -- 提问 --> G[图: 加载→策略→推理→执行→观测→校验→收尾]
    G -- ask_user/await_user<br/>终止原因=等待用户 --> U
    U -- 回答 --> G
    G -- 完成 --> U
```

图内 `终止原因=等待用户` 主动让出控制权，`SqliteSaver` 保留 `AgentState`，同 `thread_id` 重新 `graph.invoke` 恢复。

## TODO

- [ ] 多 Agent 协调
- [ ] 重写记忆系统
- [ ] 重写 Skill 载入
