

ans = []

ans.append(input())

anstr = ' '.join(ans)


with open("/home/bender/v_a/llama/BERTlog.txt", "w") as f:
    f.write(anstr)

with open("/home/bender/v_a/llama/BERTlog.txt") as f:
    print(f.read())

with open("/home/bender/v_a/llama/BERTlog.txt", "w") as f:
    f.write('')

with open("/home/bender/v_a/llama/BERTlog.txt") as f:
    print(f.read())