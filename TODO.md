# TODO - 待办事项

---

## 今日完成 (2025-03-16)

### 手臂识别改进
- [x] 简化 YOLO 关键点使用 - 只使用肩膀、肘部、手腕三点，移除髋部依赖
- [x] 添加 `calculate_shoulder_angle_simple()` 函数计算上臂与垂直方向夹角
- [x] 摄像头画面坐标系可视化 - 在关节位置绘制 X/Y/Z 轴
- [x] 四视图同时显示 - FRONT, LEFT, TOP, ISO 同时显示

### 髋部关键点处理
- [x] 实现无髋部模式：使用肩肘向量与垂直方向的夹角
- [x] 更新 `map_arm_to_qarm()` 适配新角度定义

---

## 项目待办

### Qarm 仿真器 (已完成)
- [x] 基础仿真功能
- [x] 正运动学 (FK)
- [x] 逆运动学 (IK)
- [x] 随机目标生成
- [x] 预计算可达点库
- [x] 上传到GitHub
- [x] 物理限制开关 (P键切换)
- [x] 简化版手势跟随模式

### 待优化
- [ ] 记录和回放功能
- [ ] 多机械臂仿真

### 性能优化 (已完成)
- [x] 使用局部更新代替 `fig.clf()` 重建
- [x] 简化为纯自动模式（手势跟随）

**待优化（可选）**：
- [ ] 移除 wall/ground 3D surface 绘制（`plot_surface` 很慢）
- [ ] 简化 gripper 绘制（V形夹具 → 简单线条）
- [ ] scatter 点改为 plot 线（`scatter` → `plot`）
- [ ] 降频渲染（摄像头30fps，3D视图10fps）
- [ ] 使用 blitting 技术加速 matplotlib

### 手指识别功能

**当前限制**：YOLOv8-pose 只有 17 个关键点，手腕只是一个点，无法识别手指开合

**解决方案**：集成 MediaPipe Hands
- [ ] 安装依赖：`pip install mediapipe`
- [ ] 添加 `HandTracker` 类（基于 MediaPipe Hands）
- [ ] 21 个关键点：手腕 + 5根手指各4个关节
- [ ] 计算手指弯曲度判断开合状态
- [ ] 开合状态映射到 Qarm 末端执行器

### Web 应用 (Flask)
- [ ] 完善 Flask Web 控制界面
- [ ] 局域网访问支持
- [ ] 移动端适配
- [ ] 实时手势控制集成

### 文档
- [ ] 用户使用手册
- [ ] API文档
- [ ] 演示视频

---

## Skills 待添加

- [ ] `github-publish` - 自动发布到GitHub
  - 自动创建仓库
  - 上传文件
  - 无需逐一确认权限

- [ ] `lan-share` - 局域网Web分享
  - 启动Flask应用
  - 自动获取本机IP
  - 生成访问链接
  - QR码（可选）
