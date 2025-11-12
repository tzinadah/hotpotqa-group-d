from hotpotqa_group_d.config import Env
from hotpotqa_group_d.services import parse_data

if __name__ == "__main__":
    env = Env()
    dev_fullwiki_data = parse_data()
    print(dev_fullwiki_data[0:2])
