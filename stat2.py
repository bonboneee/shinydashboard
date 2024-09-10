import numpy as np

x_values = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

probabilities = np.array([1/36, 2/36, 3/36, 4/36, 5/36, 6/36, 5/36, 4/36, 3/36, 2/36, 1/36])

# 기대값 계산
expected_value = np.sum(x_values * probabilities)

# 분산 계산
variance = np.sum((x_values - expected_value)**2 * probabilities)

# 결과 출력
print(f"기대값: {expected_value}")
print(f"분산: {variance}")

# 2X+3은?

expected_value2 = 2 * expected_value + 3

variance2 = 4 * variance  # (2X + 3)의 분산은 2^2 * X의 분산

print(f"기대값2: {expected_value2}")
print(f"분산2: {variance2}")
