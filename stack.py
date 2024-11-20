from typing import Any

class StackOverflowException(Exception):
    pass

class StackUnderflowException(Exception):
    pass

class Stack:
    def __init__(self, capacity):
        if capacity < 0:
            raise ValueError(f"capacity cannot be negative: {capacity}")
        self.capacity = capacity
        self.array = [None] * capacity
        self.top = -1
    
    def push(self, value):
        if type(value) == list:
            for v in value:
                self.push(v)
        else:
            self.top += 1
            if self.top == self.capacity:
                raise StackOverflowException()
            self.array[self.top] = value
    
    def pop(self) -> Any:
        if self.top == -1:
            raise StackUnderflowException()
        v = self.array[self.top]
        self.top -= 1
        return v
        

### Testing
import unittest

class TestStack(unittest.TestCase):
    def test_should_valueerror(self):
        with self.assertRaises(ValueError):
            Stack(-1)

    def test_should_overflow(self):
        s = Stack(2)
        with self.assertRaises(StackOverflowException):
            s.push([2, 3, 4])

    def test_should_underflow(self):
        s = Stack(5)
        with self.assertRaises(StackUnderflowException):
            s.push(1)
            s.pop()
            s.pop()

    def test_pushing_and_popping(self):
        s = Stack(5)
        s.push([2, 3])
        self.assertEqual(s.pop(), 3)
        self.assertEqual(s.pop(), 2)

if __name__ == '__main__':
    unittest.main()