"""
This data structure is used to store a queue of timestamped data. The queue has a maximum length,
and when the queue is full, the oldest data is removed to make room for new data.
The data is stored as a tuple of a timestamp and some data. The queue supports the following operations:
- enqueue(item): Add an item to the queue.
- dequeue(): Remove and return the oldest item from the queue.
- search(timestamp): Find the two items in the queue with timestamps closest to the given timestamp.
  If the given timestamp is equal to the timestamp of an item in the queue, return that item.
  If the given timestamp is less than the timestamp of the first item in the queue, return the first item.
  If the given timestamp is greater than the timestamp of the last item in the queue, return the last item.
  Otherwise, return the two items with timestamps closest to the given timestamp.
"""

from collections import deque
import bisect

class TimestampQueueElem:
    def __init__(self, timestamp, data):
        self.timestamp = timestamp
        self.data = data

    def __lt__(self, other):
        if isinstance(other, TimestampQueueElem):
            return self.timestamp < other.timestamp
        elif isinstance(other, tuple):
            return self.timestamp < other[0]
        elif isinstance(other, float):
            return self.timestamp < other
        elif isinstance(other, int):
            return self.timestamp < other
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, TimestampQueueElem):
            return self.timestamp == other.timestamp and self.data == other.data
        elif isinstance(other, tuple):
            return self.timestamp == other[0] and self.data == other[1]

    def __repr__(self):
        return f"({self.timestamp}, {self.data})"
        

class TimestampQueue:
    def __init__(self, max_length):
        self.max_length = max_length
        self.queue = deque(maxlen=max_length)

    def enqueue(self, timestamp, data):
        self.queue.append(TimestampQueueElem(timestamp, data))

    def dequeue(self):
        if self.queue:
            return self.queue.popleft()
        return None

    def search(self, timestamp):
        if len(self.queue) == 0:
            return None
        index = bisect.bisect_left(self.queue, timestamp)
        if index == len(self.queue):
            return [self.queue[len(self.queue) - 1]]
        if index == 0:
            if self.queue[0].timestamp == timestamp:
                return [self.queue[0]]
            return [self.queue[0]]
        if self.queue[index].timestamp == timestamp:
            return [self.queue[index]]
        return [self.queue[index - 1], self.queue[index]]

if __name__ == "__main__":
    # Create a TimestampQueue with max length 5
    queue = TimestampQueue(max_length=5)

    # Test enqueue
    queue.enqueue(1, 'A')
    queue.enqueue(2, 'B')
    queue.enqueue(3, 'C')
    queue.enqueue(4, 'D')
    queue.enqueue(5, 'E')
    assert list(queue.queue) == [(1, 'A'), (2, 'B'), (3, 'C'), (4, 'D'), (5, 'E')]

    # Test dequeue
    assert queue.dequeue() == (1, 'A')
    assert queue.dequeue() == (2, 'B')
    assert list(queue.queue) == [(3, 'C'), (4, 'D'), (5, 'E')]

    # Test search
    assert queue.search(3) == [(3, 'C')]
    assert queue.search(4) == [(4, 'D')]
    assert queue.search(2) == [(3, 'C')]  # (3, 'C') is the closest item to 2
    assert queue.search(6) == [(5, 'E')]
    assert queue.search(0) == [(3, 'C')]

    # Test search between two timestamps
    assert queue.search(3.5) == [(3, 'C'), (4, 'D')]
    assert queue.search(4.5) == [(4, 'D'), (5, 'E')]

    # Test max length
    queue.enqueue(6, 'F')
    queue.enqueue(7, 'G')
    queue.enqueue(8, 'H')
    assert list(queue.queue) == [(4, 'D'), (5, 'E'), (6, 'F'), (7, 'G'), (8, 'H')] # Oldest item 'C' is removed

    # Test dequeue on empty queue
    queue = TimestampQueue(max_length=3)
    assert queue.dequeue() is None