from app import model, encoder
from GPT2.sample import SampleNode

if __name__ == '__main__':
    text = "I looked into her beautiful blue eyes and said,"
    token_ids = encoder.encode(text)
    x = SampleNode(model, token_ids)

    z = x.random_new_descendents(128)
    print(encoder.decode(z.tokens(-1)))