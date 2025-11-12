import asyncio

from hotpotqa_group_d.pipelines import async_answer

if __name__ == "__main__":
    asyncio.run(async_answer("results/baseline.json"))
