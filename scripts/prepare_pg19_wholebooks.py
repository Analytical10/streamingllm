import argparse
from pathlib import Path

# 注意：移除了 concat_until_target_incremental 的导入，改用直接循环
from streamingllm_experiment.data.pg19 import iter_pg19_texts
from streamingllm_experiment.logging_utils import save_json
from streamingllm_experiment.paths import DATA_DIR
from streamingllm_experiment.tokenization import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="/data/pretrained_models/Llama-2-7b-hf")
    # 保留了 num_books，去成了 target_tokens，因为不再需要限制 token
    parser.add_argument("--num-books", type=int, default=3)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--split", default="test")
    # 将默认输出文件名调整为更通用的名字
    parser.add_argument("--output", default=str(DATA_DIR / "pg19_books.txt"))
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_name)
    texts = iter_pg19_texts(split=args.split, streaming=args.streaming)
    
    collected_texts = []
    token_count = 0
    num_used = 0
    
    print(f"开始获取并处理前 {args.num_books} 本书...")
    
    # 核心修改：忽略 token 限制，只严格根据图书数量下载和计数
    for text in texts:
        if num_used >= args.num_books:
            break
            
        print(f"正在处理第 {num_used + 1}/{args.num_books} 本书...")
        
        # 性能优化：仅对当前单本书的内容进行分词并累加 Token，避免 O(N²) 的重复计算
        book_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        token_count += book_tokens
        
        collected_texts.append(text)
        num_used += 1

    # 合并所有图书文本
    combined = "".join(collected_texts)

    # 核心修改：动态将最终的 token 总数拼接到文件名后面
    original_output_path = Path(args.output)
    # 例如：pg19_books.txt -> pg19_books_123456.txt
    output_path = original_output_path.with_name(
        f"{original_output_path.stem}_{token_count}{original_output_path.suffix}"
    )
    
    # 创建目录并写入文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(combined, encoding="utf-8")

    # 更新元数据
    meta = {
        "model_name": args.model_name,
        "actual_tokens": token_count,
        "num_books_used": num_used,
        "output": str(output_path),
    }
    save_json(output_path.with_suffix(".meta.json"), meta)
    
    # 打印最终的 token 总数与保存路径
    print(f"成功保存到: {output_path}")
    print(f"总计处理图书: {num_used} 本，Token 总数: {token_count}")


if __name__ == "__main__":
    main()