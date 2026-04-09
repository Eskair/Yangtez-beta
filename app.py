import gradio as gr
import subprocess
import os
import uuid
import shutil

def run_pipeline(file):

    try:
        # 1️⃣ 生成唯一 ID（核心）
        proposal_id = f"web_{uuid.uuid4().hex[:8]}"

        # 2️⃣ 创建独立目录
        work_dir = os.path.join("tmp", proposal_id)
        os.makedirs(work_dir, exist_ok=True)

        # 3️⃣ 保存上传文件（避免覆盖）
        file_path = os.path.join(work_dir, os.path.basename(file.name))
        shutil.copy(file.name, file_path)

        # 4️⃣ 调用后端（安全方式）
        result = subprocess.run(
            ["python", "run_review.py", "--file", file_path, "--proposal_id", proposal_id],
            check=True,
            timeout=600,
            capture_output=True,
            text=True
        )

        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)

        # 5️⃣ 找输出报告
        report_path = os.path.join(
            "src", "data", "runs", proposal_id, f"{proposal_id}_review_report.md"
        )

        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return "❌ Report not found."

    except subprocess.TimeoutExpired:
        return "⏱️ Error: Task timeout (too long)."

    except Exception as e:
        return f"❌ Error: {str(e)}"


# UI
demo = gr.Interface(
    fn=run_pipeline,
    inputs=gr.File(label="Upload PDF"),
    outputs=gr.Textbox(label="Review Result"),
    title="Yangtze AI Reviewer"
)

if __name__ == "__main__":
    demo.launch()