import mujoco as mj
import mujoco.viewer
import time
import os


# XML 모델의 정확한 경로
xml_path = os.path.join('model/spacerobot_test_mujoco_legacy.xml')

# 모델 로드
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# 시뮬레이션 실행 및 뷰어 표시
with mujoco.viewer.launch_passive(model, data) as viewer:
    while True:
        mj.mj_step(model, data)
        print(data.qpos)
        viewer.sync()
        time.sleep(0.01)
