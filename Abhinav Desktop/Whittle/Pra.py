
# Grades matrix
grades = [
    [0, 0, 0, 0],  # 0 days
    [0, 2, 1, 3],  # Course A
    [0, 2, 3, 3],  # Course B
    [0, 4, 3, 4]   # Course C
]

# Number of courses and study days
n_courses = 3
n_days = 3

# DP table initialization
dp = [[0] * (n_days + 1) for _ in range(n_courses + 1)]

# Fill the DP table
for i in range(1, n_courses + 1):  # Courses
    for j in range(0, n_days + 1):  # Days
        # Case: Skip studying this course
        dp[i][j] = dp[i-1][j]
        
        # Case: Study for 1, 2, or 3 days if possible
        if j >= 1:
            dp[i][j] = max(dp[i][j], dp[i-1][j-1] + grades[i][1])
        if j >= 2:
            dp[i][j] = max(dp[i][j], dp[i-1][j-2] + grades[i][2])
        if j >= 3:
            dp[i][j] = max(dp[i][j], dp[i-1][j-3] + grades[i][3])

# Final answer
max_grade_sum = dp[n_courses][n_days]
print("Maximum Grade Sum:", max_grade_sum)
