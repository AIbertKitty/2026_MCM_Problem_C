import pandas as pd
import os

def split_csv_by_season(input_file):
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：找不到文件 {input_file}")
        return

    try:
        # 读取CSV文件
        print(f"正在读取文件: {input_file}...")
        df = pd.read_csv(input_file)
        
        # 确保 'season' 列是数值类型，防止数据中混入字符串
        df['season'] = pd.to_numeric(df['season'], errors='coerce')

        # 1. 分割第1赛季和第2赛季
        part1 = df[(df['season'] >= 1) & (df['season'] <= 2)]
        output_file1 = 'season_1_2.csv'
        part1.to_csv(output_file1, index=False)
        print(f"已保存第1-2赛季数据至: {output_file1} (行数: {len(part1)})")

        # 2. 分割第3到27赛季
        part2 = df[(df['season'] >= 3) & (df['season'] <= 27)]
        output_file2 = 'season_3_27.csv'
        part2.to_csv(output_file2, index=False)
        print(f"已保存第3-27赛季数据至: {output_file2} (行数: {len(part2)})")

        # 3. 分割第28到34赛季
        part3 = df[(df['season'] >= 28) & (df['season'] <= 34)]
        output_file3 = 'season_28_34.csv'
        part3.to_csv(output_file3, index=False)
        print(f"已保存第28-34赛季数据至: {output_file3} (行数: {len(part3)})")

        print("\n所有文件分割完成！")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")

# 执行分割
# 请确保文件名与你实际的文件名一致
file_name = '/Users/xuyunpeng/Desktop/美赛c题2/2026_MCM_Problem_C_Data.csv'
split_csv_by_season(file_name)