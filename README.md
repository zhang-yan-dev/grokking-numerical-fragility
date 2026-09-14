# hello-python

我的第一个 Python 项目，用于学习和练习。

## 这是什么

记录我从零开始学 Python 和 Git 的过程。
用 uv 管理依赖和环境，所有代码都能直接跑。

## 运行方式

```
uv run python main.py
```

## 文件说明

- `main.py` —— 主程序
- `diag.py` —— 网络连通性诊断脚本
- `certinfo.py` —— 抓取 HTTPS 证书，用于识别中间人

## 学到的东西

- 用 uv 创建和管理 Python 项目
- Git 的三区模型：工作区 → 暂存区 → 版本库
- 分支的本质是一个指向提交的指针
- 完整的 PR 流程：建分支 → 提交 → 推送 → 开 PR → 合并
