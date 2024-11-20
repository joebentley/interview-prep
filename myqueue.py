from typing import Any

class QueueOverflowException(Exception):
    pass

class QueueUnderflowException(Exception):
    pass

class Queue:
    def __init__(self, capacity):
        if capacity < 0:
            raise ValueError(f"capacity cannot be negative: {capacity}")
        self.capacity = capacity
        self.array = [None] * capacity
        self.head = 0
        self.tail = 0
    
    def enqueue(self, value):
        if type(value) == list:
            for v in value:
                self.enqueue(v)
        else:
            if self.head == (self.tail + 1) % self.capacity:
                raise QueueOverflowException()
            self.array[self.tail] = value
            self.tail = (self.tail + 1) % self.capacity
    
    def dequeue(self) -> Any:
        if self.tail == self.head:
            raise QueueUnderflowException()
        v = self.array[self.head]
        self.head = (self.head + 1) % self.capacity
        return v
    
    def __len__(self) -> int:
        return self.tail - self.head
        

### Testing
import unittest

class TestQueue(unittest.TestCase):
    def test_should_valueerror(self):
        with self.assertRaises(ValueError):
            Queue(-1)

    def test_should_overflow(self):
        q = Queue(2)
        with self.assertRaises(QueueOverflowException):
            q.enqueue([2, 3, 4])

    def test_should_underflow(self):
        q = Queue(5)
        with self.assertRaises(QueueUnderflowException):
            q.enqueue(1)
            q.dequeue()
            q.dequeue()

    def test_enqueuing_and_queuing(self):
        q = Queue(5)
        self.assertEqual(len(q), 0)
        q.enqueue([2, 3, 4])
        self.assertEqual(len(q), 3)
        self.assertEqual(q.dequeue(), 2)
        self.assertEqual(q.dequeue(), 3)
        self.assertEqual(len(q), 1)

        q.enqueue([2, 3, 4])
        self.assertEqual(q.dequeue(), 4)
        self.assertEqual(q.dequeue(), 2)

if __name__ == '__main__':
    unittest.main()