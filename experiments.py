class StudyObject:
    def __init__(self, a: int, b: str):
        self.a = a
        self.b = b

    def __call__(self, *args, **kwargs):
        print("I am being called")


def main():
    ss = StudyObject(88, "Hello")
    ss(1, "cat")


if __name__ == '__main__':
    main()
