import torch

# 경로 설정
BASE_PATH = "/mnt/data1/intern/jeongchan/enhancer_data/data_and_model"
ATAC_CKPT = f"{BASE_PATH}/mdlm/gosai_data/binary_atac_cell_lines.ckpt"

def main():
    print(f"Loading checkpoint: {ATAC_CKPT}...")
    try:
        # 모델 전체를 로드하지 않고, 딕셔너리만 가볍게 열어서 봅니다.
        ckpt = torch.load(ATAC_CKPT, map_location="cpu")
        
        # 1. Hyper-parameters 확인
        hp = ckpt.get("hyper_parameters", {})
        
        # 2. 가능한 키워드들로 Task 이름 찾기
        # Grelu/Enformer는 보통 'tasks', 'task_names', 'target_cols' 등으로 저장합니다.
        tasks = None
        
        if "tasks" in hp:
            tasks = hp["tasks"]
        elif "task_names" in hp:
            tasks = hp["task_names"]
        elif "data_params" in hp and "tasks" in hp["data_params"]:
            tasks = hp["data_params"]["tasks"]
        
        # 3. 결과 출력
        if tasks:
            print("\n=== Found Task Names ===")
            for idx, name in enumerate(tasks):
                print(f"Index {idx}: {name}")
                
            # HepG2 찾기
            for idx, name in enumerate(tasks):
                if "hepg2" in str(name).lower():
                    print(f"\n✅ Conclusion: atac_hepg2_index should be [{idx}]")
                    break
        else:
            print("\n⚠️ Task names not found in hyper_parameters.")
            print("Keys found in hp:", list(hp.keys()))
            if "n_tasks" in hp:
                 print(f"Number of tasks: {hp['n_tasks']}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()