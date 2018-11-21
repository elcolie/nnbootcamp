import torch


class StudyObject:
    def __init__(self, a: int, b: str):
        self.a = a
        self.b = b

    def __call__(self, *args, **kwargs):
        print("I am being called")


def test():
    ss = StudyObject(88, "Hello")
    ss(1, "cat")


def main():
    output = torch.DoubleTensor([
        2.4021, 1.1024, 4.2021, 3.5821, 4.6106, 6.0499, 7.3469, 8.0217,
        8.9718, 10.0243
    ])
    prob = output.exp()
    weights = torch.multinomial(prob, 2)
    print(weights)


if __name__ == '__main__':
    main()
