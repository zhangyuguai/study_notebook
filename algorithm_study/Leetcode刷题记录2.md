# Leetcode刷题记录

## 6.前缀和

### [303. 区域和检索 - 数组不可变](https://leetcode.cn/problems/range-sum-query-immutable/)

> **一维前缀和模板**：
>
> 1. **前缀和定义**:
>
>     `s[i+1]` 表示从 `nums[0]` 到 `nums[i]` 的所有元素的和，即 `s[i+1] = s[i] + nums[i]`。`s[0]` 被初始化为 0，因此 `s[i+1]` 表示前 `i+1` 个元素的和。
>
> 2. **区间和计算**:
>
>    * 要求区间 `[left, right]` 的所有元素的和，只需用 `s[right + 1] - s[left]` 即可。
>
>    * `s[right + 1]` 表示从 `nums[0]` 到 `nums[right]` 的和。
>
>    * `s[left]` 表示从 `nums[0]` 到 `nums[left - 1]` 的和。
>
>    * 因此，`s[right + 1] - s[left]` 得到的就是从 `nums[left]` 到 `nums[right]` 的所有元素的和。

```c++
// 定义 NumArray 类来处理一维数组的前缀和
class NumArray {
public:
    // 定义一个前缀和数组 s
    vector<int> s;
    // 构造函数，初始化前缀和数组
    NumArray(vector<int>& nums) {
        // 将前缀和数组 s 的大小设为 nums 的大小加 1，并初始化为 0
        s.resize(nums.size() + 1, 0);
        // 构建前缀和数组
        for (int i = 0; i < nums.size(); i++) {
            s[i + 1] = s[i] + nums[i];
        }
    }
    // 计算区间和
    int sumRange(int left, int right) {
        // 返回区间 [left, right] 的元素和
        return s[right + 1] - s[left];
    }
};

```

### [560. 和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/)

> 思路：利用前缀和+哈希表的方式进行优化，定义为`s[i+1]`表示`[0,i]`闭区间内所有元素的和。而`s[i+1]=s[i]+a[i]`。同时要求一个闭区间`[l,r]`所有元素的和，只需用`s[r+1]-s[l]`即可。使用一个哈希表来维护前缀和出现的次数。对于给定的`k`值，我们仅需`s[r+1]-s[l]==k`即`s[l]=s[r+1]-k`只需检查是否`map`中是否存在`s[l]`对应前缀和。如果出现，加上他的次数即可。最后将当前出现的前缀和`s[r+1]`加入`map`中。
>
> 解法：
>
> 1. **前缀和数组的定义和计算**：
>
>    - 定义 `s[i + 1]` 表示从数组开始到第 `i` 个元素的累积和。
>
>    - 计算公式：`s[i + 1] = s[i] + nums[i]`。
>
> 2. **求闭区间 `[l, r]` 内所有元素的和**：
>
>    - 公式：`sum[l, r] = s[r + 1] - s[l]`。
>
>    - 这个公式通过减去左边界之前的累积和来得到指定区间的和。
>
> 3. **利用哈希表维护前缀和出现的次数**：
>
>    - 使用一个哈希表 `map` 来记录前缀和的出现次数。
>
>    - 对于每个前缀和 `s[i + 1]`，需要检查是否存在一个之前的前缀和 `s[l]` 满足 `s[r + 1] - s[l] == k`，即 `s[l] = s[r + 1] - k`。
>
>    - 如果 `s[l]` 在哈希表中存在，就把对应的次数加到结果中。
>
> 4. **更新哈希表**：
>
>    - 将当前前缀和 `s[r + 1]` 加入哈希表，并更新它的出现次数。

```c++
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        // 前缀和数组，s[i] 表示从 nums[0] 到 nums[i-1] 的累积和
        vector<int> s(nums.size() + 1, 0);
        // 哈希表，用于存储前缀和出现的次数
        unordered_map<int, int> cnt;
        // 初始化哈希表，表示前缀和为 0 出现了一次
        cnt[0] = 1;
        int res = 0;

        // 计算前缀和数组
        for (int i = 0; i < nums.size(); i++) {
            s[i + 1] = s[i] + nums[i];
        }

        // 遍历前缀和数组
        for (auto& num : s) {
            // 查找是否存在前缀和为 num - k 的情况
            if (cnt.contains(num - k)) {
                // 如果存在，则将其出现的次数加到结果中
                res += cnt[num - k];
            }
            // 更新当前前缀和的出现次数
            cnt[num]++;
        }

        return res;
    }
};

```

### [1524. 和为奇数的子数组数目](https://leetcode.cn/problems/number-of-sub-arrays-with-odd-sum/)

> 思路：利用前缀和+哈希表思路，因为此题仅需记录奇数和偶数的前缀和数目，因此可以使用两个变量来记录。
>
>  思路总结
>
> 1. **前缀和定义**：
>    - 定义前缀和数组 `s`，其中 `s[i]` 表示从数组 `arr` 开头到 `arr[i-1]` 的元素和。
>    - 通过前缀和数组，可以快速计算任意子数组的和。
> 2. **奇偶性质**：
>    - 我们需要统计和为奇数的子数组数量。可以通过观察前缀和的奇偶性质来进行统计。
>    - 如果当前前缀和为偶数，则减去一个前缀和为奇数的前缀和得到的子数组和为奇数。
>    - 同理，如果当前前缀和为奇数，则减去一个前缀和为偶数的前缀和得到的子数组和为奇数。
> 3. **计数器**：
>    - 使用两个计数器 `odd` 和 `even` 分别记录当前出现的奇数和偶数前缀和的数量。
>    - 初始化 `even` 为 1（表示空前缀和为 0，为偶数），`odd` 为 0。
> 4. **遍历数组**：
>    - 遍历数组，计算当前前缀和的奇偶性。
>    - 如果当前前缀和为奇数，则可以形成的和为奇数的子数组数量等于当前 `even` 的值。
>    - 如果当前前缀和为偶数，则可以形成的和为奇数的子数组数量等于当前 `odd` 的值。
>    - 根据当前前缀和的奇偶性更新 `odd` 和 `even` 的值。

```c++
class Solution {
public:
    int numOfSubarrays(vector<int>& arr) {
        vector<int> s(arr.size() + 1, 0);
        for (int i = 0; i < arr.size(); i++) {
            s[i + 1] = s[i] + arr[i];
        }
        long long res = 0;
        const int MOD = 1e9 + 7;
        int odd = 0, even = 1; // 初始值，前缀和为0是偶数
        for (int i = 1; i <= arr.size(); i++) {
            int temp = s[i] % 2; // 当前前缀和的奇偶性
            res += temp ? even : odd; // 如果是奇数，增加当前偶数前缀和的数量；如果是偶数，增加当前奇数前缀和的数量
            if (temp) {
                odd++; // 当前前缀和为奇数，增加奇数计数器
            } else {
                even++; // 当前前缀和为偶数，增加偶数计数器
            }
        }
        return res % MOD;
    }
};

```

### [974. 和可被 K 整除的子数组](https://leetcode.cn/problems/subarray-sums-divisible-by-k/)

> 思路：利用前缀和+哈希表的思路，如果`(s[r+1]−s[l])%k=0`即：` s[r+1]%k=s[l]%k`如果前缀和数组中两个前缀和对 `k` 取余相等，那么它们之间的子数组和就能够被 `k `整除。利用哈希表记录出现余数的次数
>
> 注意点：
>
> 1. 哈希表需要处理仅有一个元素的情况，将`{0,1}`加入哈希表中即可
> 2. 可能会出现负数，我们将取模后的数再加上`k`之后再对`k`取模即可。即`(s[i] % k + k) % k`
> 3. 前缀和数组中两个前缀和模 `k` 相等，则它们之间的子数组和能被 `k` 整除。

```c++
class Solution {
public:
    int subarraysDivByK(vector<int>& nums, int k) {
        // 前缀和数组s, s[i+1] 表示 nums[0] 到 nums[i] 的和
        vector<int> s(nums.size() + 1, 0);

        // 计算前缀和数组
        for (int i = 0; i < nums.size(); i++) {
            s[i + 1] = s[i] + nums[i];
        }

        // 哈希表 cnt 记录前缀和模 k 余数出现的次数，初始为 {0, 1}
        unordered_map<int, int> cnt{{0, 1}};

        int res = 0; // 记录符合条件的子数组数量
        for (int i = 1; i <= nums.size(); i++) {
            // 计算当前前缀和模 k 的余数，并确保余数为非负数
            int temp = (s[i] % k + k) % k;

            // 检查哈希表中是否存在当前余数，若存在则加上该余数出现的次数
            if (cnt.contains(temp)) {
                res += cnt[temp];
            }

            // 更新当前前缀和模 k 余数的出现次数
            cnt[temp]++;
        }
        
        return res; // 返回符合条件的子数组数量
    }
};

```

### [523. 连续的子数组和](https://leetcode.cn/problems/continuous-subarray-sum/)

> 思路：利用前缀和+哈希表的方法，哈希表定义为记录前缀和取模后第一次出现的位置`pos`。由于`(s[j+1]-s[i])%k==0`等价于（`s[j+1]%k==s[i]%k`）遍历前缀和数组，判断当前前缀和取模后的值是否在哈希表中存在，如果存在判断第一次出现的位置与当前位置是否至少`>=2`如果满足返回`true`。如果不存在，记录当前取模值第一次出现的位置。
>
> 注意点：
>
> 1. 要处理仅有两个元素前缀和情况，需要在`map`提前加入`{0,0}`。

```c++
class Solution {
public:
    bool checkSubarraySum(vector<int>& nums, int k) {
        // 初始化前缀和数组
        vector<int> s(nums.size() + 1, 0);
        for (int i = 0; i < nums.size(); i++)
            s[i + 1] = s[i] + nums[i];

        // 使用哈希表记录每个前缀和模 k 的余数第一次出现的位置
        unordered_map<int, int> cnt{{0, 0}};

        // 遍历前缀和数组
        for (int i = 1; i <= nums.size(); i++) {
            int mod = (s[i] % k + k) % k; // 确保余数为非负数

            // 如果当前余数已经存在于哈希表中
            if (cnt.contains(mod)) {
                // 检查当前索引 i 与记录的位置之间的距离是否大于等于 2
                if (i - cnt[mod] >= 2)
                    return true; // 找到符合条件的子数组，返回 true
            } else {
                // 记录当前余数第一次出现的位置
                cnt[mod] = i;
            }
        }
        return false; // 未找到符合条件的子数组，返回 false
    }
};

```



****

> [!NOTE]
>
> 以下为二维前缀和



---



### [304. 二维区域和检索 - 矩阵不可变](https://leetcode.cn/problems/range-sum-query-2d-immutable/)

> **二维前缀和模板**：为了避免单独处理第一行和第一列的情况，定义 `s[i+1][j+1]` 表示从 `matrix[0][0]` 到 `matrix[i][j]` 的子矩阵和。
>
> 步骤：
>
> 1.  **构建前缀和矩阵**
>
>    首先，构建一个前缀和矩阵 `s`，其中 `s[i+1][j+1]` 表示从 `matrix[0][0]` 到 `matrix[i][j]` 的子矩阵和。前缀和矩阵的构建公式为：`s[i+1][j+1]=s[i][j+1]+s[i+1][j]−s[i][j]+matrix[i][j]`
>
> 2. **计算子矩阵和**
>
>    使用前缀和矩阵，可以快速计算任意子矩阵 `[row1, col1]` 到 `[row2, col2]` 的和。
>
>    具体公式为：`sum=s[row2+1][col2+1]−s[row2+1][col1]−s[row1][col2+1]+s[row1][col1]`
>
>    * `s[row2 + 1][col2 + 1]` 表示从 `matrix[0][0]` 到 `matrix[row2][col2]` 的和。
>
>    * `s[row2 + 1][col1]` 表示从 `matrix[0][0]` 到 `matrix[row2][col1-1]` 的和，需要减去。
>
>    * `s[row1][col2 + 1]` 表示从 `matrix[0][0]` 到 `matrix[row1-1][col2]` 的和，需要减去。
>
>    * `s[row1][col1]` 表示从 `matrix[0][0]` 到 `matrix[row1-1][col1-1]` 的和，**被减去两次，需要加回来**。

```c++
class NumMatrix {
public:
    vector<vector<int>> s; // 前缀和矩阵

    // 构造函数，初始化前缀和矩阵
    NumMatrix(vector<vector<int>>& matrix) {
        int m = matrix.size(); // 矩阵的行数
        int n = matrix[0].size(); // 矩阵的列数
        s.resize(m + 1, vector<int>(n + 1, 0)); // 初始化前缀和矩阵，多一行一列用于计算前缀和

        // 计算前缀和
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                s[i + 1][j + 1] = s[i][j + 1] + s[i + 1][j] - s[i][j] + matrix[i][j];
                // s[i+1][j+1] 表示从 matrix[0][0] 到 matrix[i][j] 的子矩阵和
                // s[i][j+1] 表示加上当前行 i 的前缀和
                // s[i+1][j] 表示加上当前列 j 的前缀和
                // s[i][j] 表示去掉行 i 和列 j 的重复部分
            }
        }
    }
    // 计算子矩阵 (row1, col1) 到 (row2, col2) 的元素和
    int sumRegion(int row1, int col1, int row2, int col2) {
        return s[row2 + 1][col2 + 1] - s[row2 + 1][col1] - s[row1][col2 + 1] + s[row1][col1];
        // s[row2+1][col2+1] 表示从 (0,0) 到 (row2,col2) 的子矩阵和
        // s[row2+1][col1] 表示从 (0,0) 到 (row2,col1-1) 的子矩阵和
        // s[row1][col2+1] 表示从 (0,0) 到 (row1-1,col2) 的子矩阵和
        // s[row1][col1] 表示从 (0,0) 到 (row1-1,col1-1) 的子矩阵和
    }
};

```

## 7.差分

> [!NOTE]
>
> 以下为一维差分

### [1094. 拼车](https://leetcode.cn/problems/car-pooling/)

> 思路：使用一个数组记录每时每刻车上的人数数量`a`，由于是从`from`到`to-1`将会有`n`名乘客上车，同时`>=to`将会有`n`名乘客下车，这种对于数组区间变化，适合使用差分数组记录数组`a`中元素的变化。定义一个差分数组`diff`，由于一开始并没有乘客上车，因此`a`中元素全为`0`,差分数组初始化也为`0`。之后进行插入操作即可。最后将`diff`数组累加即可得到`a`数组。
>
> 解法：
>
> 1. **插入变化量**： 对于每个拼车请求 `[numPassengers, from, to]`：
>
>    - 在 `from` 位置增加 `numPassengers`。
>
>    - 在 `to` 位置减去 `numPassengers`（表示这些乘客在 `to` 位置下车）。
>
>    - 这可以通过在 `diff[from]` 增加 `numPassengers`，在 `diff[to]` 减少 `numPassengers` 来实现。
>
> 2. **还原实际乘客数**： 遍历差分数组 `diff`，通过累加前缀和的方式还原出每个时刻车上的总乘客数。
>
> 3. **检查是否超载**： 在还原出每个时刻的乘客总数后，检查这些数值中是否有超过容量 `capacity` 的。如果有，返回 `false`，否则返回 `true`。

```c++
class Solution {
public:
    // 插入函数，用于在差分数组中添加乘客变化量
    void insert(int l, int r, int c, vector<int> &diff) {
        diff[l] += c;
        diff[r + 1] -= c;  // 在 r+1 位置减去 c，表示乘客在 r 位置下车
    }

    // 主要函数，用于判断是否可以完成所有拼车请求而不超载
    bool carPooling(vector<vector<int>>& trips, int capacity) {
        vector<int> diff(1002, 0);  // 差分数组大小为 1002，保证不会越界

        // 遍历所有拼车请求，并在差分数组中记录乘客变化
        for (int i = 0; i < trips.size(); i++)
            insert(trips[i][1], trips[i][2] - 1, trips[i][0], diff);

        int sum = 0;

        // 通过差分数组还原出每个时刻车上的总人数的数组 a
        for (int i = 0; i < 1001; i++) {
            diff[i + 1] += diff[i];  // 累加计算每个时刻车上的乘客数
        }

        // 检查车上的最大乘客数是否超过容量
        return *max_element(diff.begin(), diff.end()) <= capacity;
    }
};

```

### [1109. 航班预订统计](https://leetcode.cn/problems/corporate-flight-bookings/)

> 思路：题中的意思，相当于对一段区间的加法，因此考虑差分数组。为了避免边界为题，我们将差分数组设置为`n+2`。最后移除第一个和最后一个元素即可。
>
> 解法：
>
> 1. 初始化差分数组 `diff`，大小为 `n + 2`，多出的两个位置用于处理边界情况。
> 2. 根据预订记录更新差分数组。
> 3. 计算前缀和还原每个航班的预订座位数。
> 4. 移除不需要的第一个和最后一个元素，得到最终的结果数组。返回结果。

```c++
class Solution {
public:
    // 插入函数，更新差分数组
    void insert(int l, int r, int c, vector<int>& diff) {
        diff[l] += c;   // 在 l 位置增加 c 个座位
        diff[r + 1] -= c; // 在 r+1 位置减少 c 个座位
    }

    vector<int> corpFlightBookings(vector<vector<int>>& bookings, int n) {
        vector<int> diff(n + 2, 0); // 初始化差分数组，大小为 n + 2

        // 遍历每个预订记录，更新差分数组
        for (int i = 0; i < bookings.size(); i++) {
            insert(bookings[i][0], bookings[i][1], bookings[i][2], diff);
        }

        // 通过差分数组计算前缀和，还原每个航班的预订座位数
        for (int i = 1; i <= n + 1; i++) {
            diff[i] += diff[i - 1];
        }

        // 移除不需要的第一个和最后一个元素，得到结果数组
        diff.pop_back();  // 移除最后一个元素
        diff.erase(diff.begin());  // 移除第一个元素

        return diff;  // 返回结果数组
    }
};
```

### [2406. 将区间分为最少组数](https://leetcode.cn/problems/divide-intervals-into-minimum-number-of-groups/)

> 思路：我们可以用差分数组来记录每个时间点上有多少区间覆盖。最终通过前缀和还原每个时间点的覆盖区间数，找到最大值即为需要的最小组数。

```c++
class Solution {
public:
    // 插入函数，更新差分数组
    void insert(int l, int r, int c, vector<int>& diff) {
        diff[l] += c;   // 在 l 位置增加 c
        diff[r + 1] -= c; // 在 r+1 位置减少 c
    }

    int minGroups(vector<vector<int>>& intervals) {
        vector<int> diff(1e6 + 100, 0); // 初始化差分数组，大小为 1e6+100
        int max_num = INT_MIN;

        // 遍历每个区间，更新差分数组
        for (int i = 0; i < intervals.size(); i++) {
            max_num = max(max_num, intervals[i][1]);
            insert(intervals[i][0], intervals[i][1], 1, diff);
        }

        // 通过差分数组计算前缀和，还原每个时间点的覆盖区间数
        for (int i = 1; i < max_num + 1; i++) {
            diff[i] += diff[i - 1];
        }
        // 返回最大覆盖数，即最小组数
        return *max_element(diff.begin(), diff.begin() + max_num + 1);
    }
};

```

### [2381. 字母移位 II](https://leetcode.cn/problems/shifting-letters-ii/)

> 思路：通过差分数组记录区间加减变动，计算前缀和得到实际的移动量，最后根据移动量更新字符串中的每个字符。这样确保字符在 'a' 到 'z' 范围内循环移动，处理了字符移动的特殊情况。
> 	注意点：
>
> 1. 本题的下标从0开始，因此在`diff`数组的更新也是从0，开始，因此我们将diff数组多设置一个空间即可

```c++
class Solution {
public:
    // 插入变动函数，用于更新差分数组
    void insert(int l, int r, int c, vector<int>& diff) {
        diff[l] += c;
        diff[r + 1] -= c;
    }

    // 主函数，执行字符串字符的移动
    string shiftingLetters(string s, vector<vector<int>>& shifts) {
        vector<int> diff(s.size() + 2, 0);  // 差分数组，大小为字符串长度加2

        // 遍历每个移动操作，更新差分数组
        for (int i = 0; i < shifts.size(); i++) {
            int c = (shifts[i][2] == 1) ? 1 : -1;  // 判断移动方向，1为向右，-1为向左
            insert(shifts[i][0], shifts[i][1], c, diff);  // 调用插入函数更新差分数组
        }

        // 计算前缀和，将差分数组转化为实际的移动量
        for (int i = 1; i <= s.size(); i++) {
            diff[i] += diff[i - 1];
        }
        diff.pop_back();  // 去掉多余的最后两位
        diff.pop_back();

        // 更新字符串中的每个字符
        for (int i = 0; i < s.size(); i++) {
            int shift = diff[i] % 26;  // 计算实际的移动量
            if (shift < 0) {  // 处理负数情况，确保字符在'a'到'z'范围内循环
                shift += 26;
            }
            s[i] = (s[i] - 'a' + shift) % 26 + 'a';  // 更新字符
        }
        return s;  // 返回更新后的字符串
    }
};

```

---

> [!NOTE]
>
> 以下为二维差分

### [2536. 子矩阵元素加 1](https://leetcode.cn/problems/increment-submatrices-by-one/)

> 二维差分矩阵模板
>
>  **思路总结：**
>
> 1. **差分矩阵初始化**
>
>    - 创建一个大小为 `(n+2) x (n+2)` 的差分矩阵 `diff`，初始值为 0。
>    - 这样做的目的是为了便于处理边界情况，避免超出数组范围。
>
> 2. **定义插入操作**
>
>    - 定义 `insert` 函数，用于在差分矩阵中更新指定区域的值。
>
>    - 接收矩形区域的左上角 `(x1, y1)` 和右下角 `(x2, y2)` 以及增加的值 `c`。
>
>    - 在差分矩阵 `diff`
>
>       中进行四个点的加减操作，以表示对矩形区域的增量：
>
>      ```c++
>      diff[x1 + 1][y1 + 1] += c;   // 在 (x1, y1) 位置增加 c
>      diff[x1 + 1][y2 + 2] -= c;   // 在 (x1, y2+1) 位置减少 c
>      diff[x2 + 2][y1 + 1] -= c;   // 在 (x2+1, y1) 位置减少 c
>      diff[x2 + 2][y2 + 2] += c;   // 在 (x2+1, y2+1) 位置增加 c
>      ```
>
> 3. **处理范围加法查询**
>
>    - 遍历所有查询，将每个查询的矩形区域加法操作插入到差分矩阵 `diff` 中。
>
> 4. **计算前缀和矩阵**
>
>    - 初始化前缀和矩阵 `res`，大小为 `(n+1) x (n+1)`，用于计算差分矩阵的前缀和。
>
>    - 前缀和矩阵 `res[i][j]` 表示从 `(0,0)` 到 `(i,j)` 的区域和。
>
>    - 计算公式为：
>
>      ```c++
>      res[i][j] = res[i - 1][j] + res[i][j - 1] - res[i - 1][j - 1] + diff[i][j];
>      ```
>
> 5. **转换为结果矩阵**
>
>    - 初始化最终结果矩阵 `temp`，大小为 `n x n`，用于存储最终的结果。
>    - 将前缀和矩阵 `res`中的值转换为原矩阵 `temp`中的值：`temp[i][j] = res[i + 1][j + 1];`
>
> 6. **返回结果矩阵**
>
>    - 最终返回处理后的结果矩阵 `temp`。

```c++
class Solution {
public:
    // 插入操作函数，在差分矩阵中更新指定区域的值
    void insert(int x1, int y1, int x2, int y2, int c, vector<vector<int>> &diff) {
        diff[x1 + 1][y1 + 1] += c;   // 在 (x1, y1) 位置增加 c
        diff[x1 + 1][y2 + 2] -= c;   // 在 (x1, y2+1) 位置减少 c
        diff[x2 + 2][y1 + 1] -= c;   // 在 (x2+1, y1) 位置减少 c
        diff[x2 + 2][y2 + 2] += c;   // 在 (x2+1, y2+1) 位置增加 c
    }

    // 处理范围加法查询的函数
    vector<vector<int>> rangeAddQueries(int n, vector<vector<int>>& queries) {
        // 初始化差分矩阵，大小为 (n+2) x (n+2)，以便处理边界
        vector<vector<int>> diff(n + 2, vector<int>(n + 2, 0));
        
        // 初始化前缀和矩阵(差分矩阵的前缀和)，大小为 (n+1) x (n+1)，以便计算原数组元素
        vector<vector<int>> res(n + 1, vector<int>(n + 1, 0));
        
        // 初始化最终结果矩阵，大小为 n x n
        vector<vector<int>> temp(n, vector<int>(n, 0));

        // 插入每个查询的区域加法操作到差分矩阵中
        for (int i = 0; i < queries.size(); i++)
            insert(queries[i][0], queries[i][1], queries[i][2], queries[i][3], 1, diff);

        // 计算前缀和矩阵，res[i][j] 为从 (0,0) 到 (i,j) 的区域和
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                res[i][j] = res[i - 1][j] + res[i][j - 1] - res[i - 1][j - 1] + diff[i][j];
            }
        }

        // 将前缀和矩阵转换为最终结果矩阵
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                temp[i][j] = res[i + 1][j + 1];
            }
        }

        // 返回最终结果矩阵
        return temp;
    }
};

```

## 8.滑动窗口

### 定长

#### [209. 长度最小的子数组](https://leetcode.cn/problems/minimum-size-subarray-sum/)

> 思路：维护一个窗口`[l,r]`处于该窗口内的元素和均是`>=tar`。题目要求找最小的子数组长度，因此，枚举右端点，当`sum>=tar`时，尝试更新左端点，让左端点尝试向右移动,如果减去左端点的值后，`sum`仍然`>=tar`则更新左端点（`sum-=nums[left++]`）.同时如果`sum>=tar`更新`ans`
>
> 解法：
>
> 1. **初始化变量**：
>
>    - `l`：左指针，初始位置为0。
>
>    - `res`：用于存储结果的最小子数组长度，初始化为`INT_MAX`。
>
>    - `sum`：当前窗口的元素和，初始化为0。
>
>    - `n`：数组的长度。
>
> 2.  **遍历数组**：
>    - 使用右指针`r`从0开始遍历数组，逐个向右移动，扩展当前窗口，并将当前元素加入窗口和中 (`sum += nums[r]`)。
>
> 3. **调整左指针**：
>    - 当 `sum`大于或等于目标值 `target`时，尝试更新左指针：
>      - 计算当前窗口的长度 `r - l + 1`，并更新结果 `res` 为两个值中较小的那个。
>      - 如果减去左端点的值后，`sum` 仍然大于或等于目标值 `target`，则更新左指针，并从窗口和中减去左端点的值 (`sum -= nums[l++]`)。
>
> 4. **返回结果**：
>    - 最后返回 `res`，如果结果没有被更新，返回0，否则返回结果的最小长度。

```c++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int l = 0, res = INT_MAX;  // 初始化左指针和结果
        int n = nums.size();
        int sum = 0;
        int r = 0;  // 初始化右指针
        for (r = 0; r < n; r++) {
            sum += nums[r];  // 右指针向右移动，增加当前窗口的和
            // 当当前窗口和减去最左边元素仍然大于等于目标值时，左指针右移
            while (sum - nums[l] >= target) {
                sum -= nums[l];  // 移除最左边元素
                l++;  // 左指针右移
            }
            // 如果当前窗口和大于等于目标值，更新结果
            if (sum >= target) {
                res = min(res, r - l + 1);  // 更新最小长度
            }
        }
        return res < INT_MAX ? res : 0;  // 如果找到有效的最小长度则返回，否则返回0
    }
};

```

#### [713. 乘积小于 K 的子数组](https://leetcode.cn/problems/subarray-product-less-than-k/)

> 思路：利用滑动窗口，遍历右端点，如果当前乘积`prode>=k`说明需要往右边移动`l`。循环结束时，`[l,r]`的乘积小于`k`，可得`[l+1,r]`乘积也小于`k`，因此总的子数组个数需要加上`(r-l+1)`。
>
> 解法：
>
> 1. **初始化**：设定左指针 `l` 和乘积 `product`。
>
> 2. **遍历数组**：使用右指针 `r` 遍历数组。
>
> 3. **调整窗口**：当 `product` 大于或等于 `k` 时，缩小窗口，确保窗口内的乘积小于 `k`。
>
> 4. **计算子数组数量**：窗口 `[l, r]` 内的子数组数量为 `r - l + 1`。

```c++
class Solution {
public:
    int numSubarrayProductLessThanK(vector<int>& nums, int k) {
        if (k <= 1) return 0;  // 如果 k <= 1，没有子数组乘积会小于 k，直接返回0
        
        int l = 0;  // 左指针初始化为0
        int product = 1;  // 窗口内元素的乘积初始化为1
        int res = 0;  // 结果变量，记录满足条件的子数组数量
        
        // 右指针从0开始遍历数组
        for (int r = 0; r < nums.size(); r++) {
            product *= nums[r];  // 将 nums[r] 乘到窗口的乘积中
            
            // 当乘积 product 大于或等于 k 时，调整左指针，缩小窗口
            while (product >= k) {
                product /= nums[l];  // 将左指针位置的元素从乘积中移除
                l++;  // 左指针右移
            }
            
            // 当乘积 product 小于 k 时，窗口内的所有子数组都满足条件
            res += (r - l + 1);  // 累加当前窗口内子数组的数量
        }
        
        return res;  // 返回最终结果
    }
};



class Solution {
public:
    int numSubarrayProductLessThanK(vector<int>& nums, int k) {
        long long prod = 1; // 用于存储当前窗口内的乘积
        int l = 0, res = 0; // 左指针初始化为0，结果初始化为0
        
        for (int r = 0; r < nums.size(); r++) {
            prod *= (long long)nums[r]; // 更新窗口内乘积
            
            // 当乘积大于等于k时，移动左指针以缩小窗口
            while (l <= r && prod >= k) {
                prod /= (long long)nums[l++];
            }
            
            // 计算以右指针结尾的子数组个数
            res += r - l + 1;
        }
        
        return res; // 返回结果
    }
};

```

#### [713. 乘积小于 K 的子数组](https://leetcode.cn/problems/subarray-product-less-than-k/)

> 思路：利用滑动窗口思想，保证一个区间内没有重复的字符，遍历右端点，采用一个哈希表记录区间内字符出现情况。
>
> 解法：
>
> 1. **字符计数**：使用哈希表 `cnt` 统计当前窗口 `[l, r]` 内各字符的出现次数。
>
> 2. **窗口调整**：当某个字符在窗口内出现次数大于1时，通过移动左指针 `l` 缩小窗口，直到窗口内没有重复字符。
>
> 3.  **长度更新**：每次右指针 `r` 移动时，更新当前无重复字符的最长子串长度。
>
> 4. **特殊情况处理**：处理字符串为空的情况，返回0。

```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char, int> cnt;  // 统计区间内字符出现的次数
        int l = 0;  // 左指针初始化为0
        int len = INT_MIN;  // 初始化最长子串长度为负无穷大

        // 右指针从0开始遍历字符串
        for (int r = 0; r < s.size(); r++) {
            cnt[s[r]]++;  // 将 s[r] 的字符计数加1

            // 当字符出现次数大于1时，调整左指针，缩小窗口
            while (cnt[s[r]] > 1) {
                cnt[s[l]]--;  // 将左指针位置的字符计数减1
                l++;  // 左指针右移
            }

            // 更新最长子串的长度
            len = max(len, r - l + 1);
        }

        // 如果 len 仍为初始值 INT_MIN，说明字符串为空，返回0；否则返回 len
        return len == INT_MIN ? 0 : len;
    }
};

```

#### [2958. 最多 K 个重复元素的最长子数组](https://leetcode.cn/problems/length-of-longest-subarray-with-at-most-k-frequency/)

> 思路：利用滑动窗口思想，维护一个窗口区间`[l,r]`，区间的元素频率均小于`k`，用一个哈希表记录区间元素出现频率。
>
> #### 解法：
>
> 1. **初始化**：设定左指针 `l` 和一个用于统计窗口内元素出现次数的哈希表 `cnt`。
> 2. **遍历数组**：使用右指针 `r` 遍历数组 `nums`。
> 3. **调整窗口**：当 `cnt[nums[r]]` 大于 `k` 时，缩小窗口，确保窗口内某个元素出现次数不超过 `k`。
> 4. **计算最大子数组长度**：每次调整窗口后，更新当前满足条件的最大子数组长度 `res`。
> 5. **返回结果**：返回满足条件的最大子数组长度。如果没有找到任何满足条件的子数组，返回0。

```c++
class Solution {
public:
    int maxSubarrayLength(vector<int>& nums, int k) {
        int l = 0;  // 左指针初始化为0
        unordered_map<int, int> cnt;  // 记录窗口内元素出现次数
        int res = INT_MIN;  // 初始化最大子数组长度为负无穷大

        // 右指针从0开始遍历数组
        for (int r = 0; r < nums.size(); r++) {
            cnt[nums[r]]++;  // 将 nums[r] 的元素计数加1

            // 当某个元素在窗口内出现次数大于k时，调整左指针，缩小窗口
            while (cnt[nums[r]] > k) {
                cnt[nums[l]]--;  // 将左指针位置的元素计数减1
                l++;  // 左指针右移
            }

            // 更新最大子数组的长度
            res = max(res, r - l + 1);
        }

        // 如果 res 仍为初始值 INT_MIN，说明没有符合条件的子数组，返回0；否则返回 res
        return res == INT_MIN ? 0 : res;
    }
};

```

#### [2730. 找到最长的半重复子字符串](https://leetcode.cn/problems/find-the-longest-semi-repetitive-substring/)

> 思路：利用滑动窗口，维护一个窗口区间，内部重复对数小于等于1，采用一个`same_cnt`记录重复子串个数，如果大于`1`，移动`l`，直到`s[l]==s[l-1]`.再取`res`和`r-l+1`最大值.
>
> ### 解法：
>
> 1. 维护一个滑动窗口 `[l, r]`，使得窗口内的字符满足“半重复”条件。
> 2. 当窗口内的连续重复字符个数超过1时，调整左指针 `l` 以缩小窗口，直到窗口内的连续重复字符个数为1。
> 3. 每次调整窗口后，更新最大窗口长度 `res`。
> 4. 最后返回 `res` 作为结果。

```c++
class Solution {
public:
    int longestSemiRepetitiveSubstring(string s) {
        int l = 0;  // 左指针初始化为0
        int res = 1;  // 初始化结果为1，因为至少有一个字符
        int same_cnt = 0;  // 记录连续重复字符的个数

        for (int r = 1; r < s.size(); r++) {  // 右指针从1开始遍历字符串
            if (s[r] == s[r - 1]) {  // 如果当前字符和前一个字符相同
                same_cnt += 1;  // 增加连续重复字符的计数
            }

            if (same_cnt > 1) {  // 当连续重复字符的个数大于1时
                l++;  // 左指针右移
                while (s[l] != s[l - 1]) {  // 继续右移左指针直到找到连续重复字符
                    l++;
                }
                same_cnt--;  // 减少连续重复字符的计数
            }

            res = max(res, r - l + 1);  // 更新最长半重复子字符串的长度
        }

        return res;  // 返回结果
    }
};

```

#### [2779. 数组的最大美丽值](https://leetcode.cn/problems/maximum-beauty-of-an-array-after-applying-operation/)

> 思路1：利用滑动窗口，只要两个区间`[x-k,x+k]`，`[y-k,y+k]`，有重叠的部分，说明能够变成相同元素。即`x+k>=y-k`即`y-x<=2*k`
>
> 思路2：利用差分数组思想，将`[x-k,x+k]`之间的变化，用差分数组表示，最后统计出最大值即可。由于最后的最大美丽数一定为数组中某个元素，因此我们可以取消负数部分，仅关注正数部分。

```c++
//滑动窗口
class Solution {
public:
    int maximumBeauty(vector<int>& nums, int k) {
        // 对数组进行排序
        sort(nums.begin(), nums.end());
        
        int l = 0;  // 初始化左指针
        int res = 0;  // 初始化结果变量，用于记录最长满足条件的子数组长度

        // 遍历数组的每个元素，右指针 r 从0到数组末尾
        for (int r = 0; r < nums.size(); r++) {
            // 当当前窗口 [l, r] 不满足条件时，移动左指针 l
            while (nums[r] - nums[l] > k * 2) {
                l++;  // 左指针右移
            }
            // 更新满足条件的最长子数组长度
            res = max(res, r - l + 1);
        }
        // 返回最终结果，即最长满足条件的子数组长度
        return res;
    }
};


//差分数组
class Solution {
public:
    int maximumBeauty(vector<int>& nums, int k) {
        // 找到数组中的最大值
        int m = *max_element(nums.begin(), nums.end());

        // 初始化差分数组，大小为 m + 2，所有元素初始为0
        vector<int> diff(m + 2, 0);

        // 遍历数组，更新差分数组
        for (int x : nums) {
            // 在 max(x - k, 0) 位置增加一个区间起点
            diff[max(x - k, 0)]++;

            // 在 min(x + k + 1, m + 1) 位置减少一个区间终点
            diff[min(x + k + 1, m + 1)]--;
        }

        int count = 0;  // 记录当前区间的元素个数
        int res = 0;  // 记录最大区间元素个数

        // 遍历差分数组，计算区间元素个数
        for (int x : diff) {
            count += x;  // 更新当前区间的元素个数
            res = max(res, count);  // 更新最大值
        }

        return res;  // 返回最大区间元素个数
    }
};

```

#### [1004. 最大连续1的个数 III](https://leetcode.cn/problems/max-consecutive-ones-iii/)

> 思路：利用滑动窗口思想，维护一个窗口`[l,r]`，窗口内部包含`0`的个数`<=k`。当`>k`时应该进行收缩左边界。最后取`res`和`r-l+1`最大值。
>
> ### 关键点总结
>
> 1. **滑动窗口和双指针**：动态维护一个有效的窗口。
> 2. **统计窗口内 `0` 的数量**：通过表达式 `count += 1 - nums[r]` 计算窗口内 `0` 的数量。
> 3. **调整窗口以满足条件**：当 `0` 的数量超过 `k` 时，通过移动左边界缩小窗口。
> 4. **记录和更新结果**：在每次右指针移动时更新最大子数组长度。

```c++
class Solution {
public:
    int longestOnes(vector<int>& nums, int k) {
        int res = 0;  // 用于存储最长的子数组长度
        int l = 0;  // 滑动窗口的左边界
        int count = 0;  // 滑动窗口内 0 的数量

        // 遍历数组，右指针从头到尾依次遍历
        for (int r = 0; r < nums.size(); r++) {
            count += 1 - nums[r];  // 如果是 0，count 加一；如果是 1，count 不变

            // 如果 count 超过了 k，收缩左边界
            while (count > k) {
                count -= 1 - nums[l];  // 将 l 向右移动，并减少 count 中的 0 的数量
                l++;
            }

            // 更新最长的子数组长度
            res = max(res, r - l + 1);
        }

        // 返回最长的子数组长度
        return res;
    }
};

```

#### :pig: [2962. 统计最大元素出现至少 K 次的子数组](https://leetcode.cn/problems/count-subarrays-where-max-element-appears-at-least-k-times/)

> 思路1：利用滑动窗口，此题的思路是，固定左端点，移动右端点，当`count`满足条件时候，说明[l,r]满足条件，那么同理有`[l,r+1],[l,r+2]`满足条件。当前条件下满足条件的总数为`n-r`。加上当前满足条件总数。移动`l`指针。
>
> 思路2：同样利用滑动窗口，此题为当`r`移动到满足要求位置时，移动`l`到不满足要求为止，数组中满足要求个数为`l`。
>
> 解法2：
>
> 1. **扩展窗口**：右指针 `r` 从左到右遍历整个数组。
>    - 如果当前元素是最大值，则增加 `count`。
>
> 2. **收缩窗口**：当窗口内最大值的出现次数正好等于 `k` 时：
>    - 移动左指针 `l` 向右，直到窗口内最大值的出现次数少于 `k`。
>
> ### 关键点
>
> 1. **滑动窗口**：使用双指针技术动态调整窗口的大小。
> 2. **计数器**：通过 `count` 统计窗口内最大值的出现次数，判断窗口是否符合条件。
> 3. **子数组计数**：每次发现一个符合条件的窗口时，将所有以 `r` 为右端点的子数组数量加到 `res` 中。
>
> ### 关键点2
>
> 1. **滑动窗口**：使用双指针技术动态调整窗口的大小。
> 2. **计数器**：通过 `count` 统计窗口内最大值的出现次数，判断窗口是否符合条件。
> 3. **子数组计数**：每次发现窗口内最大值的出现次数等于 `k` 时，调整左指针并统计以当前右指针位置为结束的所有符合条件的子数组数量。

```c++
class Solution {
public:
    long long countSubarrays(vector<int>& nums, int k) {
        int l = 0; // 左指针初始化为数组的起始位置
        long long res = 0; // 初始化结果为0
        int count = 0; // 用于统计当前窗口内最大值的出现次数
        int max_num = *max_element(nums.begin(), nums.end()); // 找到数组中的最大值

        for (int r = 0; r < nums.size(); r++) { // 右指针遍历整个数组
            // 如果当前元素是最大值，则增加计数
            count += (nums[r] == max_num ? 1 : 0);
            
            // 当窗口内最大值的出现次数达到或超过k
            while (count >= k) {
                // 增加符合条件的子数组数目
                res += nums.size() - r;
                // 移动左指针前减少计数
                count -= (nums[l] == max_num ? 1 : 0);
                l++; // 左指针右移
            }
        }
        return res; // 返回最终结果
    }
};

//解法2
class Solution {
public:
    long long countSubarrays(vector<int>& nums, int k) {
        int l = 0; // 左指针初始化为数组的起始位置
        long long res = 0; // 初始化结果为0
        int count = 0; // 用于统计当前窗口内最大值的出现次数
        int max_num = *max_element(nums.begin(), nums.end()); // 找到数组中的最大值

        for (int r = 0; r < nums.size(); r++) { // 右指针遍历整个数组
            // 如果当前元素是最大值，则增加计数
            count += (nums[r] == max_num ? 1 : 0);
            
            // 当窗口内最大值的出现次数正好等于k时，移动左指针
            while (count == k) {
                // 移动左指针前减少计数
                count -= (nums[l] == max_num ? 1 : 0);
                l++; // 左指针右移
            }
            
            // 统计以当前右指针位置为结束的所有符合条件的子数组
            res += l;
        }
        return res; // 返回最终结果
    }
};


```

#### :japanese_ogre: [1658. 将 x 减到 0 的最小操作数](https://leetcode.cn/problems/minimum-operations-to-reduce-x-to-zero/)

> 思路1：利用双指针+前缀和+后缀和解题。先计算后缀和，也就是全部移除最后一个元素时，最多能到达位置。再枚举前缀和，不断地加上前缀，如果大于了`x`。说明需要减少后缀，因此，后指针向后移动，如果过程中`sum>x`说明已经不能使`x`减少到零了，退出循环。当`sum==x`时，计算最小的步数，分别是前缀`l+1`，以及后缀的`n-r`。返回和`res`比较最小的一个
>
> 思路2：反向思考，要求x减到0的最小操作数，等价于求使得最长子数组和为`sum-x`。利用滑动窗口，维护一个窗口`[l,r]`。寻找使得子数组和为`sum-x`。记录子数组的最大长度。最后返回`n-res`

```c++
//正向解，双指针，计算前缀和与后缀和。
class Solution {
public:
    int minOperations(vector<int>& nums, int x) {
        int sum = 0;
        int n = nums.size();
        int r = n;

        // 计算最长的后缀和
        while (r > 0 && sum + nums[r - 1] <= x) {
            sum += nums[--r];
        }
        if (r == 0 && sum < x) return -1; // 全部移除也无法满足要求

        int res = sum == x ? n - r : n + 1;

        // 移动左指针，计算前缀和
        for (int l = 0; l < n; l++) {
            sum += nums[l];
            while (r < n && sum > x) {
                // 说明当前已经超出了，需要 r 往后面移动
                sum -= nums[r++];
            }
            if (sum > x) break; // 缩小失败，说明前缀过长
            if (sum == x) {
                res = min(res, l + 1 + n - r); // 更新最小操作数
            }
        }

        return res > n ? -1 : res;
    }
};
//反向解题
class Solution {
public:
    int minOperations(vector<int>& nums, int x) {
        int target = accumulate(nums.begin(), nums.end(), 0) - x;
        // 计算数组的总和，并计算目标值 target = sum - x
        // 寻找一个最长的子数组，使得其和等于 target

        if (target < 0)
            return -1; // 如果 target 小于 0，说明无法找到这样的子数组

        int res = -1, left = 0, sum = 0, n = nums.size();
        // res 用于记录满足条件的最长子数组长度，初始值为 -1 表示尚未找到
        // left 为滑动窗口的左指针，sum 用于记录当前窗口内的元素和，n 为数组长度

        for (int right = 0; right < n; right++) {
            sum += nums[right]; // 将当前元素加入窗口
            while (sum > target) 
                sum -= nums[left++]; // 当窗口内元素和超过 target 时，移动左指针缩小窗口
            if (sum == target)
                res = max(res, right - left + 1); // 如果窗口内元素和等于 target，更新 res
        }

        return res == -1 ? -1 : n - res;
        // 如果没有找到满足条件的子数组，返回 -1
        // 否则返回总长度减去最长子数组长度，表示最少移除元素数
    }
};

```

#### [76. 最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/)

> 思路：使用滑动窗口+哈希表的方式，我们枚举 s 子串的右端点 right（子串最后一个字母的下标），如果子串涵盖 t，就不断移动左端点 left 直到不涵盖为止。在移动过程中更新最短子串的左右端点。
>
> ### 思路
>
> 1. **初始化字符频率计数器**:
>    - 使用两个数组 `cnt_s` 和 `cnt_t` 分别记录字符串 `s` 和 `t` 中字符的频率。
>    - 使用 `less` 记录 `t` 中还未满足的字符种类数。
> 2. **滑动窗口的双指针法**:
>    - 右指针 `r` 遍历字符串 `s`，将字符加入窗口并更新 `cnt_s`。
>    - 如果当前字符的频率达到 `t` 中的要求，则减少 `less`。
>    - 当 `less` 为 0 时，说明当前窗口包含了所有 `t` 中的字符。
> 3. **缩小窗口**:
>    - 尝试通过移动左指针 `l` 缩小窗口，同时更新结果。
>    - 如果窗口中字符频率减少到不满足 `t` 的要求，则增加 `less` 并继续扩展右指针。
> 4. **记录最小窗口**:
>    - 记录最小窗口的起始位置 `ans_left` 和结束位置 `ans_right`。
> 5. **返回结果**:
>    - 如果找到了满足条件的窗口，返回对应的子串；否则返回空字符串。
>
> 

```c++
class Solution {
public:
    // 判断 cnt_s 是否覆盖 cnt_t 中的字符要求
    string minWindow(string s, string t) {
        int m = s.size(); // 字符串 s 的长度
        int ans_left = -1, ans_right = m, l = 0; // 初始化最优解区间
        int less = 0; // 记录 t 中未满足的字符种类数
        int cnt_s[128] = {0}, cnt_t[128] = {0}; // 字符频率计数器

        // 记录 t 中每个字符的频率，并初始化 less
        for (char ch : t)
            less += cnt_t[ch]++ == 0;

        // 滑动窗口右指针 r 遍历字符串 s
        for (int r = 0; r < m; r++) {
            cnt_s[s[r]]++;
            // 如果当前字符频率达到 t 中的要求，减少 less
            if (cnt_s[s[r]] == cnt_t[s[r]])
                less--;

            // 当 less 为 0 时，说明当前窗口包含了所有 t 中的字符
            while (less == 0) {
                if (r - l < ans_right - ans_left) { // 更新最小窗口
                    ans_left = l;
                    ans_right = r;
                }
                // 如果移除左边界字符后，窗口不再满足条件，增加 less
                if (cnt_s[s[l]] == cnt_t[s[l]])
                    less++;
                cnt_s[s[l++]]--; // 移动左指针
            }
        }
        // 如果没有找到满足条件的窗口，返回空字符串
        return ans_left == -1 ? "" : s.substr(ans_left, ans_right - ans_left + 1);
    }
};

```

#### [1052. 爱生气的书店老板](https://leetcode.cn/problems/grumpy-bookstore-owner/)

> 思路:定长滑动窗口，统计不生气时的满意度，加上最大的一直生气时满意度减去区间内不生气满意度的差值。
>
> ## 解法：
>
> 1. **定长滑动窗口**：窗口的大小固定为 `minutes`。
> 2. **统计不生气时的满意度**：遍历整个数组，累加老板不生气时的客户数量到 `good_sum`。
> 3. **滑动窗口内生气时的满意度**：计算滑动窗口内的客户数量 `sum`，其中老板生气时的客户数量加到 `sum`。
> 4. **减去窗口内不生气时的满意度**：在滑动窗口内，更新 `max_sum` 为生气时的最大客户数量。
> 5. **计算总满意度**：将 `good_sum` 和 `max_sum` 相加，得到最大化满意的客户数量。

```c++
class Solution {
public:
    int maxSatisfied(vector<int>& customers, vector<int>& grumpy, int minutes) {
        int sum = 0, max_sum = 0, good_sum = 0;
        int l = 0;
        
        // 遍历整个customers数组
        for(int r = 0; r < customers.size(); r++) {
            // 计算不生气时的客户满意度总和
            good_sum += grumpy[r] ? 0 : customers[r];
            
            // 计算当前窗口内，老板生气时的客户满意度
            sum += grumpy[r] ? customers[r] : 0;
            
            // 确保窗口大小为 minutes
            if(r < minutes - 1)
                continue;
            
            // 更新最大的一直生气时的客户满意度
            max_sum = max(sum, max_sum);
            
            // 从左边界移出元素，减去左边界客户的满意度（如果老板当时生气）
            sum -= grumpy[l] ? customers[l] : 0;
            
            // 左指针右移
            l++;
        }
        
        // 返回不生气时的满意度总和加上生气时的最大满意度
        return good_sum + max_sum;
    }
};

```

#### [1423. 可获得的最大点数](https://leetcode.cn/problems/maximum-points-you-can-obtain-from-cards/)

> 思路：正向思考，先选择数组后缀和能够最大取得数,然后再在前缀和进行滑动窗口，同时更新答案，取最大值。
>
> ### 解法1
>
> 1. **计算后缀和**：
>    - 从数组末尾开始，计算长度为 `k` 的后缀和 `total_sum`。
>    - 例如，对于数组 `cardPoints` 和 `k = 3`，后缀和将包括最后 `3` 个元素的和。
> 2. **计算前缀和并滑动窗口**：
>    - 初始化 `prefix_sum` 为 `0`。
>    - 遍历前缀的每一个元素，从 `0` 到 `k-1`。
>    - 对于每个前缀元素，更新 `prefix_sum` 和 `total_sum`，并计算新的 `total_sum`，即从后缀和中减去相应的元素，加上当前前缀元素。
>    - 更新最大得分 `max_score`。
>
> 思路2：反向思考，题目要求求在开头或者结尾抽取卡牌点数的最大值，那么可以转换为找到一个长度为`n-k`的连续子数组，使得该子数组的和最小。即最后的结果为==数组总和-子数组最小和==
>
> ### 解法2：
>
> 1. **问题转换**：将问题转换为找到一个长度为 `n-k` 的连续子数组，使得该子数组的和最小。这样最终的结果就是数组总和减去该子数组的最小和。
> 2. **计算总和**：计算整个数组的总和 `total_sum`。
> 3. **滑动窗口**：使用滑动窗口方法遍历数组，寻找长度为 `n-k` 的子数组的最小和 `min_subarray_sum`。
>    - 初始化滑动窗口的和 `current_sum` 为 0。
>    - 遍历数组，当窗口大小达到 `n-k` 时，更新最小和，并调整窗口（滑动窗口）。
> 4. **计算结果**：最后结果为 `total_sum - min_subarray_sum`。

```c++
//法1
class Solution {
public:
    int maxScore(vector<int>& cardPoints, int k) {
        int l=0,sum=0,res=0,r=cardPoints.size();
        //计算最多可以在后缀和中拿的点数。
        for(;cardPoints.size()-r<k;r--)
            sum+=cardPoints[r-1];
        res=sum;
        for (int i = 0; i < k; i++) {
            sum += cardPoints[i] - cardPoints[r + i];
            res = max(res, sum);
        }
        return res;
    }
};

//法2
class Solution {
public:
    // 寻找连续的大小为tar的子数组，使得其和最小
    int maxScore(vector<int>& cardPoints, int k) {
        int tar = cardPoints.size() - k;  // 需要找的子数组的长度
        int sum_ = accumulate(cardPoints.begin(), cardPoints.end(), 0);  // 整个数组的总和
        int l = 0, sum = 0, res = INT_MAX;  // 初始化左指针，当前子数组的和，结果（最小的子数组和）
        
        // 如果 k 等于数组长度，直接返回总和，因为需要选择所有的卡牌
        if (cardPoints.size() == k)
            return sum_;
        
        // 使用滑动窗口寻找长度为tar的子数组的最小和
        for (int r = 0; r < cardPoints.size(); r++) {
            sum += cardPoints[r];  // 当前元素加入窗口
            if (r < tar - 1)  // 当窗口大小还不到 tar 时，继续扩大窗口
                continue;
            res = min(res, sum);  // 更新最小和
            sum -= cardPoints[l++];  // 左指针右移，窗口大小保持不变
        }
        
        // 最终结果为总和减去最小的子数组和
        return sum_ - res;
    }
};

```

#### [2134. 最少交换次数来组合所有的 1 II](https://leetcode.cn/problems/minimum-swaps-to-group-all-1s-together-ii/)

> 思路：将数组看为一个循环数组，如何保证下标能够进行循环呢？将数组长度扩大一倍即可（指的是，`l,r`,可以遍历的范围。但需要对数组长度取余即`(r+n)%n`）。计算出数组中`1`的总数。即为滑动窗口的大小。在移动窗口过程中，找到窗口中`1`最多的位置。答案即为`sum-count`。在遍历的过程中不断的更新答案。`res=min(res,sum-count)`
>
> ### 总结思路
>
> 1. **初始化变量**：
>    - `count`：当前滑动窗口中 `1` 的数量。
>    - `l`：滑动窗口的左指针。
>    - `sum`：数组中 `1` 的总数，计算滑动窗口的大小。
>    - `res`：记录最少的交换次数，初始值设为一个较大值 `INT_MAX`。
>    - `r`：滑动窗口的右指针。
> 2. **遍历数组**：
>    - 遍历长度为两倍的数组，通过 `r % nums.size()` 实现循环数组效果。
>    - 每次遇到 `1`，增加 `count`。
>    - 当窗口大小达到 `sum` 时，计算当前窗口中的 `1` 的数量，并更新 `res` 为 `sum - count` 的最小值。
>    - 调整左指针 `l` 并根据左指针位置的元素调整 `count`。
> 3. **返回结果**：
>    - 返回最少交换次数 `res`。
>    
>    思路2：变成一个循环数组，那么直接长度变为两倍即可，统计数组中1的个数，即为滑动窗口的大小，窗口内部的1肯定不会被交换，窗口内部的0，需要使用窗口外的1进行交换。因此交换的次数 为 sum-s  因此更新`res=min(res,sum-s)`

```c++
class Solution {
public:
    int minSwaps(vector<int>& nums) {
        int count = 0;  // 当前窗口中1的数量
        int l = 0;  // 滑动窗口的左指针
        int sum = accumulate(nums.begin(), nums.end(), 0);  // 数组中1的总数，滑动窗口大小
        int res = INT_MAX;  // 记录最少交换次数
        int r;

        for (r = 0; r < nums.size() * 2; r++) {
            if (nums[r % nums.size()] == 1)  // 窗口右侧进入1
                count++;
            
            if (r < sum - 1)  // 窗口长度不足sum时，继续扩展窗口
                continue;

            res = min(res, sum - count);  // 更新最小交换次数

            if (nums[l % nums.size()] == 1)  // 窗口左侧离开1
                count--;
            
            l++;  // 左指针右移
        }

        return res;  // 返回最少交换次数
    }
};


class Solution {
public:
    int minSwaps(vector<int>& nums) {
        vector<int> temp(nums.begin(), nums.end());
        for (int x : nums)
            temp.push_back(x);
        int sum = accumulate(nums.begin(), nums.end(), 0);
        int l = 0;
        int res = 0x3f3f3f3f;
        int s = 0;
        for(int r=0;r<temp.size();r++){
            s+=temp[r];

            if(r<sum-1)continue;
            res=min(res,sum-s);
            s-=temp[l++];
        }
        return res;
    }
};

```

#### [2653. 滑动子数组的美丽值（计数排序待解决）](https://leetcode.cn/problems/sliding-subarray-beauty/)

> 思路：利用滑动窗口，然后计数排序计算第k小的值。
>
> ### 思路
>
> 1. **定义计数数组**：因为 `nums` 的取值范围在 `-50` 到 `50` 之间，因此使用一个长度为 `101` 的计数数组来记录每个数字出现的次数。我们用 `nums[i] + 50` 作为计数数组的索引，以处理负数。
> 2. **滑动窗口**：使用滑动窗口方法，每次向右移动窗口，将新的元素加入计数数组，并从窗口中移除最左边的元素。
> 3. **查找第 `x` 小的值**：在每次窗口移动后，通过遍历计数数组，找出当前窗口中第 `x` 小的值，并记录到结果数组中。
> 4. **优化**：只有当窗口大小达到 `k` 时，才开始计算并记录结果。

```c++
class Solution {
public:
    vector<int> getSubarrayBeauty(vector<int>& nums, int k, int x) {
        vector<int> count(2 * 50 + 1, 0); // 计数数组，范围为 -50 到 50，偏移量为 50
        vector<int> res(nums.size() - k + 1, 0); // 结果数组，大小为 nums.size() - k + 1
        int l = 0; // 窗口左边界

        // 遍历数组
        for (int r = 0; r < nums.size(); r++) {
            count[nums[r] + 50]++; // 增加计数
            if (r < k - 1) // 如果窗口大小不足 k，继续扩展窗口
                continue;

            // 查找当前窗口中的第 x 小的值
            int left = x;
            for (int j = 0; j < 50; j++) {
                left -= count[j];
                if (left <= 0) {
                    res[r - k + 1] = j - 50; // 记录结果
                    break;
                }
            }
            
            count[nums[r - k + 1] + 50]--; // 移除窗口最左边的元素
        }
        return res;
    }
};

```

#### [1297. 子串的最大出现次数](https://leetcode.cn/problems/maximum-number-of-occurrences-of-a-substring/)

> 思路：本题的`maxsize`是一个多余条件，因为，如果答案为长度为`maxsize`的串，那么长度为`minsize`也符合答案。因此我们仅需要维护一个窗口的大小为`minsize`的窗口即可。保证窗口内部满足出现字符种类小于`maxLetters`。
>
> ### 关键点
>
> 1. 当字符出现次数为0后需要从map中移除。

```c++
class Solution {
public:
    int maxFreq(string s, int maxLetters, int minSize, int maxSize) {
        int res=0;
        unordered_map<string,int > cnt;
        unordered_map<char,int> cntch;
        int l=0;
        int n=s.size();
        
        for(int r=0;r<n;r++){
            cntch[s[r]]++;//统计窗口内部字符种类出现次数
            if(r<minSize-1)continue;//未达到窗口大小
            if(cntch.size()>maxLetters){//如果窗口内字符种类个数超过了最大限制，需要移动左指针，直到满足要求。必须固定窗口大小，也就是说，此处只能用if不能使用while  要保证左指针移动一次右指针也移动一次。
                if(--cntch[s[l]]==0)/
                    cntch.erase(s[l]);
                l++;
                continue;
            }
            string temp=s.substr(l,minSize);
            cnt[temp]++;
            if(--cntch[s[l]]==0)
                cntch.erase(s[l]);
            l++;
            res=max(res,cnt[temp]);
        }
        return res;

    }
};
```

#### [438. 找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/)

> 思路：利用哈希表+滑动窗口。由题可知，异位词的长度一定是固定的。因此我们采用滑动窗口，利用哈希表来统计字符的频率，要求字符的频率完全相同，因此我们需要两个哈希表。由于字母全是小写字符。因此我们可以使用两个长度为26的数组来代替哈希表。
>
> #### 关键点
>
> 1. 题目中有子串的信息，优先思考能否使用滑动窗口。

```c++
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        vector<int> res;
        
        int len=p.size();
        vector<int> cnt(27,0);
        vector<int> cnt_p(27,0);

        for(char ch:p)cnt_p[ch-'a']++;
        int l=0;
        for(int r=0;r<s.size();r++){
            cnt[s[r]-'a']++;
            if(r<len-1)continue;
            bool flag=true;
            for(int i=0;i<=26;i++){
                if(cnt[i]!=cnt_p[i]){
                    flag=false;
                    break;
                }
            }
            if(flag)res.push_back(l);
            cnt[s[l]-'a']--;
            l++;
        }
        return res;
    }
};
```

******

> [!NOTE]
>
> 以下为不定长滑动窗口

### 不定长

#### [1493. 删掉一个元素以后全为 1 的最长子数组](https://leetcode.cn/problems/longest-subarray-of-1s-after-deleting-one-element/)

> 思路：利用不定长滑动窗口思想，统计窗口内部`1`以及`0`的个数，当`0`的个数大于`1`后，移动`l`指针，直到`0`的个数`<=1`。最后不断的更新`res`.由于必须要删除一个元素。因此如果`0`的个数等于`1`。那么需要删除一个`1`的个数。
>
> ### 解法
>
> 1. **滑动窗口**：使用滑动窗口的技巧，遍历数组。维护一个窗口，其中最多包含一个 `0`。
> 2. **窗口内计数**：在窗口内统计 `0` 和 `1` 的数量。当窗口内 `0` 的数量超过 1 时，移动左边界缩小窗口，直到窗口内 `0` 的数量不超过 1。
> 3. **更新结果**：在每次窗口移动过程中，更新包含最多 `1` 的最长子数组的长度。

```c++
class Solution {
public:
    int longestSubarray(vector<int>& nums) {
        int res = 0; // 最长子数组长度
        int l = 0; // 窗口左边界
        int count_0 = 0; // 窗口内 0 的数量
        int count_1 = 0; // 窗口内 1 的数量

        // 遍历数组
        for (int r = 0; r < nums.size(); r++) {
            count_0 += nums[r] ? 0 : 1; // 如果当前元素是 0，增加 count_0
            count_1 += nums[r] ? 1 : 0; // 如果当前元素是 1，增加 count_1

            // 如果窗口内 0 的数量超过 1，移动左边界缩小窗口
            while (count_0 > 1) {
                count_0 -= nums[l] ? 0 : 1; // 如果左边界元素是 0，减少 count_0
                count_1 -= nums[l] ? 1 : 0; // 如果左边界元素是 1，减少 count_1
                l++; // 移动左边界
            }

            // 更新结果，子数组长度为 count_1，减去一个 1 是因为允许删除一个元素
            res = max(res, count_0 == 0 ? count_1 - 1 : count_1);
        }

        return res;
    }
};

```

#### [904. 水果成篮](https://leetcode.cn/problems/fruit-into-baskets/)

> 思路：利用滑动窗口+哈希表，哈希表记录，窗口内不同种类个数。如果超出题目所给种类要求，需要移动左指针进行减少种类。更新`res=max(res,r-l+1)`
>
> ### 解法：
>
> 1. **初始化**：
>    - `res` 用于存储最终结果，即最长满足条件的子数组长度。
>    - `l` 是滑动窗口的左边界，初始值为 0。
>    - `cnt` 是一个哈希表，记录当前窗口内不同类型水果的数量。
> 2. **遍历数组**：
>    - 使用 `r` 作为窗口的右边界，遍历整个水果数组。
>    - 将 `r` 所指向的水果类型加入哈希表 `cnt` 中，并增加其数量。
> 3. **调整窗口**：
>    - 如果哈希表 `cnt` 中水果类型的数量超过两种，移动左边界 `l` 收缩窗口。
>    - 减少 `l` 所指向水果的数量，如果某种水果的数量减为 0，从哈希表中删除该水果类型。
> 4. **更新结果**：
>    - 每次调整窗口后，计算当前窗口长度，并更新 `res`，保持 `res` 为当前最长的满足条件的子数组长度。

```c++
class Solution {
public:
    int totalFruit(vector<int>& fruits) {
        int res = 0; // 最终结果，记录最长子数组长度
        int l = 0; // 滑动窗口左边界
        unordered_map<int, int> cnt; // 记录窗口内不同类型水果的数量
        
        // 遍历水果数组
        for (int r = 0; r < fruits.size(); r++) {
            cnt[fruits[r]]++; // 将右边界的水果加入哈希表
            
            // 当窗口内的水果种类超过两种时，收缩窗口
            while (cnt.size() > 2) {
                // 减少左边界水果的数量
                if (--cnt[fruits[l]] == 0) 
                    cnt.erase(fruits[l]); // 数量为0则删除该水果类型
                l++; // 移动左边界
            }
            
            // 更新结果，计算当前窗口长度
            res = max(res, r - l + 1);
        }
        
        return res; // 返回最长子数组长度
    }
};

```

#### [1695. 删除子数组的最大得分](https://leetcode.cn/problems/maximum-erasure-value/)

> 思路：采用不定长滑动窗口+哈希表做法。利用哈希表记录窗口中相同元素出现的次数。如果`>1`需要不断的移动`l`，直到相同元素出现次数`<=1`。同时在滑动的过程中记录窗口内总和`sum`。不断的更新`res=max(res,sum)`。
>
> ### 解法：
>
> 1. **滑动窗口**：
>
>    * 使用不定长的滑动窗口来维护一个子数组，该子数组中的元素都是唯一的。
>
>    - 滑动窗口的右边界不断向右移动，将新的元素加入窗口。
>
> 2. **哈希表**：
>
>    - 使用哈希表 `cnt` 记录窗口中每个元素出现的次数。
>    - 当新元素加入窗口后，如果哈希表中该元素的计数超过1（即出现重复元素），就需要缩小窗口。
>
>
> 4. **缩小窗口**：
>    - 通过移动窗口的左边界 `l`，逐步移除左边界的元素，直到窗口中不再有重复元素。
>    - 在缩小窗口的过程中，需要更新当前窗口内的和 `cursum`。
>
>
> 6. **更新结果**：
>     - 在每次调整窗口后，计算当前窗口内的和，并更新最大和 `res`，保持 `res` 为当前最大和的子数组。
>

```c++
class Solution {
public:
    int maximumUniqueSubarray(vector<int>& nums) {
        unordered_map<int, int> cnt; // 记录每个数字出现的次数
        int l = 0, res = 0; // 初始化左边界和结果
        int cursum = 0; // 当前子数组的和
        
        for (int r = 0; r < nums.size(); r++) {
            cursum += nums[r]; // 增加右边界数字的值到当前子数组和
            cnt[nums[r]]++; // 增加右边界数字的计数
            
            // 如果当前数字出现超过一次，则缩小左边界，直到没有重复数字
            while (cnt[nums[r]] > 1) {
                cnt[nums[l]]--; // 减少左边界数字的计数
                cursum -= nums[l]; // 从当前子数组和中减去左边界数字的值
                l++; // 移动左边界
            }
            
            // 更新最大子数组和
            res = max(res, cursum);
        }
        
        return res; // 返回结果
    }
};

```

### 

> 思路：利用滑动窗口，分为两种情况进行计算，一种为最长为`T`，另一种为最长为`F`。取二者的最大值
>
> ### 解法：
>
> - 使用滑动窗口技术来找到最大长度的连续子字符串，允许最多 `k` 个 `T` 或 `F` 被替换。
> - 分别计算允许替换 `T` 和 `F` 的情况下的最大子字符串长度。
> - 返回两者的最大值。

```c++
class Solution {
public:
    // 辅助函数，用于计算允许替换特定字符情况下的最大子字符串长度
    int maxConsecutiveChars(string& answerKey, int k, char ch) {
        int count = 0;
        int l = 0;
        int max_len = 0;

        // 使用滑动窗口技术
        for (int r = 0; r < answerKey.size(); r++) {
            if (answerKey[r] == ch)
                count++;

            // 窗口中的特定字符超过k个时，移动左指针
            while (count > k) {
                if (answerKey[l++] == ch)
                    count--;
            }

            // 更新最大长度
            max_len = max(max_len, r - l + 1);
        }

        return max_len;
    }

    // 主函数
    int maxConsecutiveAnswers(string answerKey, int k) {
        // 计算允许替换 'T' 和 'F' 的情况下的最大子字符串长度
        int max_t = maxConsecutiveChars(answerKey, k, 'T');
        int max_f = maxConsecutiveChars(answerKey, k, 'F');

        // 返回两者的最大值
        return max(max_t, max_f);
    }
};

```

#### [1438. 绝对差不超过限制的最长连续子数组](https://leetcode.cn/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/)

> 思路：使用不定长滑动窗口，同时使用数据结构维护最大值，最小值（`multi_set`）。
>
> ### 解法：
>
> 1. 维护一个不定长滑动窗口`[l,r]`
> 2. 使用一个`multi_set`维护窗口内部最大值和最小值。
> 3. 如果最大值和最小值差值`>limit`需要移动`l`指针直到符合要求`<=limit`
>    1. 中途不断更新`res=max(res,r-l+1)	`

```c++
class Solution {
public:
    int longestSubarray(vector<int>& nums, int limit) {
        multiset<int, greater<int>> cache; // 用于存储窗口内的元素，并保持有序
        int l = 0; // 左指针
        int res = 0; // 结果，最长子数组的长度

        for (int r = 0; r < nums.size(); r++) {
            cache.insert(nums[r]); // 将新元素加入有序集合
            int sub = *cache.begin() - *cache.rbegin(); // 计算当前窗口内最大值和最小值的差值

            // 当差值超过 limit 时，移动左指针并调整集合
            while (sub > limit) {
                cache.erase(cache.find(nums[l])); // 从集合中移除左指针指向的元素
                l++; // 移动左指针
                sub = *cache.begin() - *cache.rbegin(); // 重新计算差值
            }

            res = max(res, r - l + 1); // 更新最长子数组的长度
        }

        return res; // 返回结果
    }
};

```

#### [2401. 最长优雅子数组](https://leetcode.cn/problems/longest-nice-subarray/)

> 思路：如果一个数`x`与另一个数`y`==按位与`(&)`==的值为`0`，另一个数与他们`(|)`==按位或==的值相与为`0`，那么同时也和数`x`数`y`==按位与==为`0`。我们利用这个思路，取一个`sum`表示子数组按位与
>
> ### 关键点：
>
> 1. 使用按位或 (`|`) 来维护当前窗口的所有元素。
> 2. 使用按位与 (`&`) 来检查当前元素是否能加入窗口。
> 3. 使用按位异或 (`^`) 来从窗口中移除元素。
> 4. 滑动窗口保证窗口内所有元素的按位与结果为0。
> 5. 通过移动左指针，维护窗口的“Nice”性质。

```c++
class Solution {
public:
    int longestNiceSubarray(vector<int>& nums) {
        int sum = nums[0];  // 当前窗口内所有元素的按位或结果，初始为第一个元素
        int l = 0;          // 左指针，初始为0
        int res = 1;        // 结果，最长“Nice”子数组的长度，初始为1

        // 右指针从第二个元素开始遍历数组
        for (int r = 1; r < nums.size(); r++) {
            // 如果当前窗口内有元素与 nums[r] 按位与结果不为0，则移动左指针
            while ((sum & nums[r]) != 0) {
                sum ^= nums[l++];  // 从当前窗口的按位或结果中移除左指针指向的元素
            }
            sum |= nums[r];  // 将 nums[r] 加入到当前窗口的按位或结果中
            res = max(res, r - l + 1);  // 更新最长“Nice”子数组的长度
        }

        return res;  // 返回结果
    }
};

```

#### [1234. 替换子串得到平衡字符串（第一次未ac）](https://leetcode.cn/problems/replace-the-substring-for-balanced-string/)

> 思路：利用滑动窗口，维护一个滑动窗口`[l,r]`，如果窗口外部的每个字母个数`>n/4`则说明无法替换。否则的话说明可以替换。窗口的长度即为最小的替换字符串。`l`不断往右移动。同时更新最小值
>
> ### 思路总结
>
> 1. **初始化和统计字符频率**：
>    - 使用哈希表 `cnt` 统计初始字符串中每个字符的频率。
>    - 计算平衡字符串中每个字符的目标频率 `m`。
> 2. **检查初始字符串是否已经平衡**：
>    - 如果初始字符串中每个字符的频率都等于目标频率 `m`，则直接返回0，因为字符串已经是平衡的。
> 3. **滑动窗口遍历字符串**：
>    - 使用右指针 `r` 遍历字符串，对每个字符频率减1。
>    - 使用内层 `while` 循环调整左指针 `l`，使得窗口内所有字符频率不超过目标频率 `m`。
>    - 在调整过程中，更新最小窗口长度 `res`。

```c++
class Solution {
public:
    int balancedString(string s) {
        unordered_map<char, int> cnt; // 用于统计字符频率的哈希表
        int l = 0, res = INT_MAX;     // 左指针初始化为0，结果初始化为最大整数
        int m = s.size() / 4;         // 每个字符在平衡字符串中的目标频率

        // 统计初始字符串中每个字符的频率
        for (char ch : s)
            cnt[ch]++;

        // 如果初始字符串已经平衡，则返回0
        if (cnt['Q'] == m && cnt['W'] == m && cnt['E'] == m && cnt['R'] == m)
            return 0;

        // 使用滑动窗口遍历字符串
        for (int r = 0; r < s.size(); r++) {
            cnt[s[r]]--; // 减少右指针当前字符的频率

            // 如果当前窗口内所有字符频率不超过目标频率，则调整左指针并更新结果
            while (cnt['Q'] <= m && cnt['W'] <= m && cnt['E'] <= m && cnt['R'] <= m) {
                res = min(res, r - l + 1); // 更新最小窗口长度
                cnt[s[l]]++;               // 增加左指针当前字符的频率
                l++;                       // 移动左指针
            }
        }

        return res; // 返回最小窗口长度
    }
};

```



#### [1658. 将 x 减到 0 的最小操作数](https://leetcode.cn/problems/minimum-operations-to-reduce-x-to-zero/)

> 思路：本题要求删除的数之和为x，同时每次只能从数组的两边进行删除数字。因此，我们可以从反面考虑，题目转换为，找到一个最长的子数组之和为sum-x ，则反过来就是删除最小的元素使得数组两边的值为x。采用不定长滑动窗口

```c++
class Solution {
public:
    int minOperations(vector<int>& nums, int x) {
        long long sum=accumulate(nums.begin(),nums.end(),0);
        long long tar=sum-x;
        if(tar<0)return -1;
        int l=0;
        int res=-1;//由于最后可能删除整个数组。因此会导致res不会更新。因此要单独的进行判断。
        long long temp_sum=0;
        for(int r=0;r<nums.size();r++){
            temp_sum+=nums[r];
            while(temp_sum>tar){
                temp_sum-=nums[l++];
            }
            if(temp_sum==tar)
                res=max(res,r-l+1);
        }
        
        return res>=0?nums.size()-res:-1;
    }
};
```



#### [1838. 最高频元素的频数](https://leetcode.cn/problems/frequency-of-the-most-frequent-element/)

> 思路：本题要求最大的频率，可以观察到，元素的位置与结果无关，因此我们可以先将数组进行排序，之后，翻译题目要求为：子窗口内其余元素与最大元素之间的差值之和不能超过k，因此我们可以维护一个滑动窗口。同时用sum记录窗口内的和。差值为：`(r-l+1)*nums[r]-sum`。如果差值大于了k，那我们需要移动l指针。`sum-=nums[l++]`。最后更新res即可。

```c++
class Solution {
public:
    int maxFrequency(vector<int>& nums, int k) {
        sort(nums.begin(), nums.end());  // 排序
        
        int l = 0;  // 左指针
        int res = 1;  // 最小频率至少为 1
        long long sum = 0;  // 当前窗口内元素的和
        
        for (int r = 0; r < nums.size(); r++) {
            sum += nums[r];  // 增加右指针指向的元素到 sum
            
            // 如果将窗口内所有元素变为 nums[r] 所需的操作数超过了 k
            while ((long long)(r - l + 1) * nums[r] - sum > k) {
                sum -= nums[l++];  // 收缩窗口
            }

            // 更新最大频率
            res = max(res, r - l + 1);
        }

        return res;
    }
};

```



#### [1234. 替换子串得到平衡字符串](https://leetcode.cn/problems/replace-the-substring-for-balanced-string/)

> 思路：利用哈希表+滑动窗口，利用哈希表统计整个数组中每个字符的出现次数。只有让滑动窗口外面的所有字符个数都小于等于tar。才能够替换成功。因此，在满足条件的时候，我们需要收缩窗口来找到最小的满足条件的窗口。
>
> #### 关键点
>
> 1. 需要提前判断一下是否不用替换就满足条件。不然的话，会导致l指针不断的移动，最终导致越界访问。

```c++
class Solution {
public:
    int balancedString(string s) {
        int tar=s.size()/4;
        unordered_map<char,int> cnt;

        for(char ch:s)cnt[ch]++;
        int l=0;
        int res=0x3f3f3f3f;
        if(cnt['Q']==tar&&cnt['R']==tar&&cnt['W']==tar&&cnt['E']==tar)return 0;
        for(int r=0;r<s.size();r++){
            cnt[s[r]]--;
            //当窗口外的满足条件全部小于等于tar
            while(cnt['Q']<=tar&&cnt['R']<=tar&&cnt['W']<=tar&&cnt['E']<=tar){
                res=min(res,r-l+1);
                cnt[s[l++]]++;
            }
        }
        return res;
    }
};
```



******

> [!NOTE]
>
> 以下为求子数组个数

### 子数组个数

#### [2799. 统计完全子数组的数目](https://leetcode.cn/problems/count-complete-subarrays-in-an-array/)

> 思路：滑动窗口+哈希表记录，利用一个`set`先统计原数组中不同元素个数。用一个map`统计`窗口内部每种元素出现次数。直到`map`中不同元素数量等于`n`时，移动左指针。当不满足要求时候。加上左指针（子数组左端点 < left 的都是合法的）
>
> ### 思路总结
>
> 1. **初始化集合和哈希表**：
>    - 使用 `set<int>` 存储数组中所有不同的元素。
>    - 使用 `unordered_map<int, int>` 记录滑动窗口中各元素出现的次数。
> 2. **计算不同元素的总数**：
>    - 通过集合的大小 `n`，获取数组中不同元素的总个数。
> 3. **滑动窗口遍历数组**：
>    - 使用右指针 `r` 遍历数组，记录窗口中每个元素的出现次数。
>    - 使用内层 `while` 循环调整左指针 `l`，确保窗口中包含所有不同元素。
>    - 在调整过程中，减少左指针当前元素的出现次数，如果次数为0，则从哈希表中移除。
>    - 每次找到一个包含所有不同元素的窗口时，累加左指针位置 `l` 到结果 `res`，因为从 `0` 到 `l` 的所有子数组都是完整子数组。

```c++
class Solution {
public:
    int countCompleteSubarrays(vector<int>& nums) {
        set<int> old_arr_set; // 用于存储数组中的不同元素
        unordered_map<int, int> cnt; // 记录窗口中各元素出现的次数
        
        // 将数组中的所有不同元素插入到集合中
        for (int num : nums)
            old_arr_set.insert(num);
        
        int n = old_arr_set.size(); // 数组中不同元素的总个数
        int l = 0, res = 0; // 左指针初始化为0，结果初始化为0
        
        // 使用滑动窗口遍历数组
        for (int r = 0; r < nums.size(); r++) {
            cnt[nums[r]]++; // 记录右指针当前元素的出现次数
            
            // 如果窗口中包含所有不同元素
            while (cnt.size() == n) {
                if (--cnt[nums[l++]] == 0) // 减少左指针当前元素的出现次数，如果次数为0，则从哈希表中移除
                    cnt.erase(nums[l - 1]);
            }
            
            res += l; // 更新结果，累加左指针位置，因为从0到l的所有子数组都是完整子数组
        }
        
        return res; // 返回结果
    }
};

```

#### [1358. 包含所有三种字符的子字符串数目](https://leetcode.cn/problems/number-of-substrings-containing-all-three-characters/)

> 思路：利用滑动窗口+哈希表统计。维护不定长窗口`[l,r]`。其中如果当满足条件`cnt.size()>=3`一直移动左端点。直到不满足条件。此时子数组左端点 < left 的都是合法的。即加上左端点的值`res+=l`
>
> ### 思路总结
>
> 1. **初始化变量**：
>    - `l` 为左指针，初始值为 0。
>    - `res` 为结果，用于记录满足条件的子字符串数量，初始值为 0。
>    - `n` 为字符串的长度。
>    - `cnt` 为哈希表，用于记录窗口内字符的频次。
> 2. **遍历字符串**：
>    - 使用右指针 `r` 遍历字符串，逐个将字符加入窗口并更新其频次。
> 3. **调整窗口**：
>    - 每次将字符加入窗口后，检查窗口是否包含所有三种字符（`a`、`b`、`c`）。
>    - 当窗口内包含所有三种字符时，缩小窗口，从左指针开始移除字符，更新左指针 `l` 并减少窗口内字符的频次。
> 4. **统计子字符串**：
>    - 每次调整窗口后，统计以当前左指针到右指针范围内的所有子字符串的数量，具体为 `l`，因为以当前右指针 `r` 为右边界的所有满足条件的子字符串数量为从左指针到当前右指针的位置的长度。

```c++\
class Solution {
public:
    int numberOfSubstrings(string s) {
        int l = 0, res = 0, n = s.size();
        unordered_map<char, int> cnt;

        for (int r = 0; r < n; r++) {
            cnt[s[r]]++; // 将右指针指向的字符计入窗口
            
            // 当窗口内包含所有三种字符时，计算满足条件的子字符串数量
            while (cnt.size() == 3) {
                res += n - r; // 计算以 r 为右边界的满足条件的子字符串数量
                if (--cnt[s[l++]] == 0) // 从窗口中移除左指针指向的字符，并移动左指针
                    cnt.erase(s[l - 1]);
            }
        }
        return res;
    }
};


class Solution {
public:
    int numberOfSubstrings(string s) {
        int l = 0, res = 0, n = s.size();
        unordered_map<char, int> cnt;

        for (int r = 0; r < n; r++) {
            cnt[s[r]]++; // 将右指针指向的字符计入窗口
            
            // 当窗口内包含所有三种字符时，调整窗口大小
            while (cnt.size() >= 3) {
                if (--cnt[s[l++]] == 0) // 从窗口中移除左指针指向的字符，并移动左指针
                    cnt.erase(s[l - 1]);
            }
            
            // 每次窗口调整后，计算左指针到当前右指针的所有子字符串
            res += l;
        }
        return res;
    }
};

```

#### [2537. 统计好子数组的数目](https://leetcode.cn/problems/count-the-number-of-good-subarrays/)

> 思路：利用不定长滑动窗口+哈希表。用哈希表记录窗口内部`[l,r]`相同元素出现个数，用`count`记录窗口配对数个数。同时如果配对数个数`>=k`不断的移动左端点直到不满足条件。此时`left`以及`<left`的端点均满足条件。更新`res+=l`.
>
> 配对数计算：不断的累加，2个增加1个，3个增加2个，4个增加3个，5个增加4个。

```c++
class Solution {
public:
    long long countGood(vector<int>& nums, int k) {
        int count = 0; // 记录当前滑动窗口内的配对数
        long long res = 0; // 结果，满足条件的子数组数量
        int l = 0; // 滑动窗口的左边界
        unordered_map<int, int> cnt; // 记录滑动窗口内每个数字的出现次数
        
        for (int r = 0; r < nums.size(); r++) { // 遍历数组，r 为右边界
            count += cnt[nums[r]]; // 将当前数字加入滑动窗口时，增加相应的配对数
            cnt[nums[r]]++; // 更新哈希表，记录当前数字的出现次数
            
            // 当配对数大于等于 k 时，调整左边界
            while (count >= k) {
                cnt[nums[l]]--; // 移出左边界的数字，并更新哈希表
                count -= cnt[nums[l++]]; // 减少相应的配对数，并移动左边界
            }
            
            res += l; // 将当前左边界 l 加入结果，表示以 r 为右边界的所有满足条件的子数组数量
        }
        
        return res; // 返回结果
    }
};

```

#### [2962. 统计最大元素出现至少 K 次的子数组](https://leetcode.cn/problems/count-subarrays-where-max-element-appears-at-least-k-times/)

> 思路：对于求子数组的有关的，想到滑动窗口，假设区间[l,r]符合条件，那么[l,r+1],[l,r+2]…..[l,len]必然也符合条件。因此遇到符合条件的。应该设置为答案加上 len-r  即 res+=len-r   即所有的符合条件的相加。可以转化为 符合条件的时候，移动左指针，直到不符合条件。此时的从 0,1,2,3,4,….,l-1 都是符合条件的。因此我们只需要将 sum(l)即可。

```c++
class Solution {
public:
    long long countSubarrays(vector<int>& nums, int k) {
        
        int l=0;
        long long res=0;
        int max_num=*max_element(nums.begin(),nums.end());
        int cnt=0;
        for(int r=0;r<nums.size();r++){
            cnt+=nums[r]==max_num?1:0;
            while(cnt>=k){
                cnt-=nums[l++]==max_num?1:0;
            }
            res+=l;
        }
        return res;
    }
};
```

#### [930. 和相同的二元子数组](https://leetcode.cn/problems/binary-subarrays-with-sum/)

> 思路：对于这种恰好等于的题目，转换为  ==x  —>  >=x  减去>=(x+1)  将滑动窗口提取为函数，使用f(x)-f(x+1)

```c++
class Solution {
public:
    int fun(vector<int>& nums, int x) {
        int l = 0;
        int res = 0;
        long long sum = 0;
        for (int r = 0; r < nums.size(); r++) {
            sum += nums[r];  // 扩大窗口
            // 收缩窗口，直到 sum < x
            while (sum >= x&&l<=r) {
                sum -= nums[l++];
            }
            // l 位置是窗口的左边界，窗口中的所有子数组都符合条件
            res += l;
        }
        return res;
    }

    int numSubarraysWithSum(vector<int>& nums, int goal) {
        return fun(nums, goal) - fun(nums, goal + 1);  // 用差值得到和为 goal 的子数组个数
    }
};

```



## 9.单调队列

### [239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/)

> 思路：使用单调队列进行处理，本题要求滑动窗口内最大值，因此我们使用单调递减的滑动窗口，队头即为滑动窗口内的最大值。**队列中元素为数组中下标**
>
> ### 关键点
>
> 1. 当队头下标如果超出窗口需要移除队头
> 2. 如果队尾元素`<=`当前元素，移除队尾元素。原因是我们维护的是一个单调递减的数组，**队尾元素一定是>当前元素的**
> 3. 如果达到了窗口大小，将队头元素加入结果数组

```c++
class Solution {
public:
    static const int N = 1e5 + 10; // 定义一个常量N，表示队列的最大长度
    int front = 0, rear = -1; // 初始化队列的头部和尾部指针
    int q[N]; // 用于存储元素下标的队列

    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> res; // 用于存储结果的数组

        for (int i = 0; i < nums.size(); i++) {
            // 移除不在当前滑动窗口范围内的下标
            while (front <= rear && q[front] < i - k + 1) {
                front++;
            }

            // 移除队列中所有小于当前元素的元素，以保持队列的单调递减性质
            while (front <= rear && nums[q[rear]] <= nums[i]) {
                rear--;
            }

            // 将当前元素的下标添加到队列
            q[++rear] = i;

            // 从第 k 个元素开始，将当前窗口的最大值加入结果数组
            if (i >= k - 1) {
                res.emplace_back(nums[q[front]]);
            }
        }

        return res; // 返回结果数组
    }
};

```

### [862. 和至少为 K 的最短子数组](https://leetcode.cn/problems/shortest-subarray-with-sum-at-least-k/)

> 思路：利用前缀和数组，能够快速的计算子数组的和，由于本题有负数，因此前缀和不一定是单调递增的，我们要保证用一个队列保证前缀和一定是单调递增的，只有这样，我们在遍历的过程中，用`s[r]-s[q[front]]`所得到的数组长度才是最小的。定义`s[r+1]=s[r]+nums[r]`。那么`s[r]-s[l]`代表的含义即为开区间数组即`[l,r)`。数组的长度为`r-l`。

```c++
#include <vector>
#include <climits>
using namespace std;

class Solution {
public:
    static const int N = 1e5 + 10; // 队列最大大小
    int front = 0, rear = -1; // 队列的前后指针
    int q[N]; // 自定义队列的数组

    // 思路：利用前缀和和自定义队列优化滑动窗口，记录最短的长度
    int shortestSubarray(vector<int>& nums, int k) {
        int n = nums.size();
        vector<long long> s(n + 1, 0); // 前缀和数组
        int res = INT_MAX; // 初始化结果为最大值

        // 初始化前缀和数组
        for (int i = 0; i < n; ++i) {
            s[i + 1] = s[i] + nums[i];
        }

        for (int r = 0; r <=n; ++r) {
            // 更新最短子数组长度
            while (front <= rear && s[r] - s[q[front]] >= k) {
                res = min(res, r  - q[front]);
                front++;
            }

            // 保持队列的单调递增
            while (front <= rear && s[q[rear]] >= s[r]) {
                rear--;
            }

            // 将当前索引入队
            q[++rear] = r;
        }

        return res == INT_MAX ? -1 : res; // 如果没有找到符合条件的子数组，返回 -1
    }
};


```

### [1438. 绝对差不超过限制的最长连续子数组](https://leetcode.cn/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/)

> 思路1：采用滑动窗口+可以排序的容器（map/muilset），记录窗口内最大值和最小值，如果不符合条件，那么移动左指针，直到符合条件位置。
>
> 思路2：采用滑动窗口+两个单调队列（一个保存最大值，一个保存最小值）

```c++
#include <vector>
#include <deque>
using namespace std;

class Solution {
public:
    static const int N = 1e5 + 10;
    int front = 0, rear = -1, q_up[N], q_down[N], front_up = 0, rear_up = -1;
    
    int longestSubarray(vector<int>& nums, int limit) {
        int res = 0;
        int l = 0;
        
        // 维护两个单调队列
        for (int r = 0; r < nums.size(); r++) {
            while (front <= rear && q_down[rear] < nums[r]) rear--; // 递减队列
            while (front_up <= rear_up && q_up[rear_up] > nums[r]) rear_up--; // 递增队列

            q_down[++rear] = nums[r];
            q_up[++rear_up] = nums[r];

            // 当队列中的最大值和最小值的差大于 limit 时，移动左边界
            while (q_down[front] - q_up[front_up] > limit) {
                if (nums[l] == q_down[front]) front++;
                if (nums[l] == q_up[front_up]) front_up++;
                l++;
            }

            res = max(res, r - l + 1);
        }
        return res;
    }
};
```

### [2398. 预算内的最多机器人数目](https://leetcode.cn/problems/maximum-number-of-robots-within-budget/)

> 思路：采用单调队列+前缀和+滑动窗口来解题，使用滑动窗口不断的移动，单调队列使用单调递减队列，维护窗口内的最大值。对于窗口内部能够快速的计算出最大值，以及窗口的和。

```c++
class Solution {
public:
    int maximumRobots(vector<int>& chargeTimes, vector<int>& runningCosts, long long budget) {
        int n = chargeTimes.size();
        vector<long long> sum(n + 1, 0);
        deque<int> q;
        
        // 计算前缀和
        for(int i = 0; i < n; i++) {
            sum[i + 1] = sum[i] + runningCosts[i];
        }

        int res = 0;
        int l = 0;

        for(int r = 0; r < n; r++) {
            // 维护单调递减队列
            while(!q.empty() && chargeTimes[q.back()] <= chargeTimes[r]) {
                q.pop_back(); 
            }
            q.push_back(r);

            // 检查窗口是否满足条件
           
            while(l <= r && (chargeTimes[q.front()] + (r - l + 1) * (sum[r + 1] - sum[l])) > budget) {
                if(l >= q.front()) {  // 原代码中条件写反了
                    q.pop_front();
                }
                l++;
            }
            
            res = max(res, r - l + 1);
        }
        return res;
    }
};
```



## 9.单调栈



### [739. 每日温度](https://leetcode.cn/problems/daily-temperatures/)

> 思路：利用单调栈，从左到右，维护一个递增的序列，如果当前栈顶元素小于当前元素，那么说明大于栈顶元素第一个位置已经找到，即为`i-st[idx]`.随后栈顶元素出栈，并更新`res`.
>
> ### 关键点
>
> 1. 使用单调栈维护一个递增序列。
>
> 2. 当遇到比栈顶元素大的温度时，更新结果数组并弹出栈顶元素。

```c++
class Solution {
public:
    static const int N=1e5+10;
    int st[N],idx=-1;
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        //实现一个单调栈
        vector<int> res(temperatures.size(),0);
        for(int i=0;i<temperatures.size();i++){
            //如果栈中元素小于当前元素，立即更新,并出栈
            while(idx>=0&&temperatures[st[idx]]<temperatures[i]){
                int pos=st[idx--];
                res[pos]=i-pos;
            }
            st[++idx]=i;
        }
        return res;
    }
};
```

### [1475. 商品折扣后的最终价格](https://leetcode.cn/problems/final-prices-with-a-special-discount-in-a-shop/)

> 思路：单调栈的应用，遇到当前栈顶元素在数组中的值如果大于等于price[i]。说明找到第一个下标，在原数组进行更新即可。

```c++
#include <vector>
using namespace std;

class Solution {
public:
    static const int N = 510; // 栈的最大容量，根据题目输入的最大限制设定
    int st[N], idx = -1; // 初始化栈和栈顶指针

    vector<int> finalPrices(vector<int>& prices) {
        for (int i = 0; i < prices.size(); i++) {
            // 如果当前价格小于等于栈顶元素对应的价格，更新栈顶元素价格并弹出栈顶元素
            while (idx >= 0 && prices[st[idx]] >= prices[i]) {
                prices[st[idx]] = prices[st[idx]] - prices[i]; // 更新栈顶元素价格
                idx--; // 弹出栈顶元素
            }
            st[++idx] = i; // 当前元素索引压入栈
        }
        return prices; // 返回更新后的价格数组
    }
};

```

### [496. 下一个更大元素 I](https://leetcode.cn/problems/next-greater-element-i/)

> 思路：利用单调栈+哈希表，哈希表记录nums2中每个数的下一个最大数，最后进行构造res数组

```c++
#include <vector>
#include <map>
using namespace std;

class Solution {
public:
    static const int N = 1010; // 定义栈的最大容量
    int st[N], idx = -1; // 初始化栈和栈顶指针
    
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
        vector<int> res(nums1.size(), -1); // 初始化结果数组
        map<int, int> cnt; // 记录nums2中每个元素的下一个更大元素
        
        // 遍历nums2，填充cnt
        for (int i = 0; i < nums2.size(); i++) {
            // 如果栈不为空且当前元素比栈顶元素大，更新栈顶元素的下一个更大元素
            while (idx >= 0 && nums2[st[idx]] < nums2[i]) {
                cnt[nums2[st[idx]]] = nums2[i];
                idx--; // 弹出栈顶元素
            }
            st[++idx] = i; // 当前元素的索引压入栈
        }
        
        // 遍历nums1，根据cnt中的信息构造结果数组
        for (int i = 0; i < nums1.size(); i++) {
            res[i] = cnt[nums1[i]];
        }
        
        // 处理没有下一个更大元素的情况
        for (int i = 0; i < nums1.size(); i++) {
            res[i] = res[i] ? res[i] : -1;
        }
        
        return res; // 返回结果数组
    }
};

```

### [503. 下一个更大元素 II](https://leetcode.cn/problems/next-greater-element-ii/)

> 思路：采用单调栈+处理循环数组。一个元素一定不会访问两次，那么我们仅需要在第一次遇见的时候入栈即可。当栈顶元素对应元素小于当前元素，立即更新res。并且栈顶元素出栈。

```c++
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
    static const int N = 1e4 + 100; // 定义栈的最大容量
    int st[N], idx = -1; // 初始化栈和栈顶指针

    vector<int> nextGreaterElements(vector<int>& nums) {
        //unordered_map<int, int> cnt; // 记录每个元素的下一个更大元素
        int n = nums.size();
        vector<int> res(n,-1);
        // 遍历数组两次以处理循环数组的情况
        for (int i = 0; i < 2 * n; i++) {
            int current = nums[i % n]; // 循环处理数组
            
            // 更新当前栈顶元素的下一个更大元素
            while (idx >= 0 && nums[st[idx]] < current) {
                res[st[idx]] = current;
                idx--;
            }
            
            // 只在第一次遇到该元素时，才将其压入栈中
            if (i < n) {
                st[++idx] = i;
            }
        }
        
        return res;
    }
};

```

### [1019. 链表中的下一个更大节点](https://leetcode.cn/problems/next-greater-node-in-linked-list/)

> 思路：利用单调栈，同时此题有两种思路，第一种是从左到右进行遍历，第二种是从右到左进行遍历。
>
> 1. 从左到右进行遍历，栈中存放的为单调递增的序列**下标**。每次栈中元素对应的值小于当前元素，更新栈中元素。同时采用一个res数组进行保存栈中每个元素的值。res数组的长度即为当前要添加元素的下标。
> 2. 从右到左进行遍历，栈中存放为单调递减的序列，采用递归实现即可，如果栈中元素大于当前元素，更新当前元素，并将其入栈。如果栈中元素小于等于当前元素，那么栈中元素出栈。

```c++
//思路1
class Solution {
public:
    vector<int> nextLargerNodes(ListNode *head) {
        vector<int> ans;
        stack<int> st; // 单调栈（只存下标）
        for(auto cur=head;cur;cur=cur->next){
            while(!st.empty()&&ans[st.top()]<cur->val){
                ans[st.top()]=cur->val;
                st.pop();
            }
            st.push(ans.size());
            ans.push_back(cur->val);
    }
    while(!st.empty()){
        ans[st.top()]=0;
        st.pop();
    }
        return ans;
    }
};

//思路2
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    static const int N=1e4+100;
    int st[N],idx=-1;
    vector<int> res;
    void dfs(ListNode *head){
        if(!head)return ;
        dfs(head->next);
        while(idx>=0&&st[idx]<=head->val)
            //如果栈顶元素小于等于当前元素，出栈。
            idx--;
        if(idx>=0)
        res.push_back(st[idx]);
        else
        res.push_back(0);
        st[++idx]=head->val;
        
    }

    vector<int> nextLargerNodes(ListNode* head) {
        dfs(head);
        reverse(res.begin(),res.end());
        return res;
    }
};

```

### [962. 最大宽度坡](https://leetcode.cn/problems/maximum-width-ramp/)

>思路：维护一个左端点的单调递减序列，如果使用单调递增序列，那么栈顶的元素一定不会比栈底的元素更优。不需要进行出栈，仅仅维护左端点进栈即可。从尾部开始遍历数组，寻找区间的右端点。如果满足`nums[st.top()]<=nums[i]`即尝试更新答案，并出栈

```c++
class Solution {
public:
    int maxWidthRamp(vector<int>& nums) {
        int res=0;
        stack<int> st;
        //维护一个单调递减栈
        for(int i=0;i<nums.size();i++){
            if(st.size()==0||nums[st.top()]>nums[i])st.push(i);
        }
       //从右边开始遍历区间右端点
       for(int i=nums.size()-1;i>=0;i--){
            while(st.size()&&nums[st.top()]<=nums[i]){
                int j=st.top();
                res=max(res,i-j);
                st.pop();
            }
       }
        return res;
    }
};
```

### [853. 车队](https://leetcode.cn/problems/car-fleet/)

> 思路：利用排序+单调栈，首先，将每个车，进行排序，一共有两种实现方式，从左到右单调栈，以及从右到左单调栈，然后，计算每个车的到达时间。如果是从小到大排序那么，要从左到右进行单调递减栈。如果Atime<Btime 同时B能够到达终点，那么A一定能够追上B。**我们思考在栈中保存，车队的队头车**也就是time消耗最大的车。最后栈的大小即为车队的数量。

```c++
class Solution {
public:
    typedef pair<int, int> PII;

    int carFleet(int target, vector<int>& position, vector<int>& speed) {
        int n = position.size();
        vector<PII> cars(n);
        
        // 将每辆车的位置和速度配对
        for (int i = 0; i < n; i++) {
            cars[i] = {position[i], speed[i]};
        }

        // 按照位置从大到小排序
        sort(cars.rbegin(), cars.rend());
        //sort(cars.begin(), cars.end());

        // 计算每辆车的到达时间
        vector<double> times(n);
        for (int i = 0; i < n; i++) {
            times[i] = (double)(target - cars[i].first) / cars[i].second;
        }

        // 使用栈来模拟车队的形成
        stack<double> st;
        int fleets = 0;

        for (int i = n-1; i >= 0; i--) {
           while(st.size()&&times[i]>=st.top())st.pop();
           st.push(times[i]);
        }
        
        /*
        for (int i = 0; i < n; i++) {
           while(st.size()&&times[i]>=st.top())st.pop();
           st.push(times[i]);
        }*/
        return st.size();
    }
};
```



## 9.字符串匹配

### [28. 找出字符串中第一个匹配项的下标](https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/)

> 思路：`kmp`模板题，牢记`next`数组含义，代表模式串以当前下标结尾的子串的最长公共前后缀长度。同时，给出定义，主串与模式串都在前面加一个哨兵空格，可以避免回退时`-1` `next`数组长度对应也要`+1` `i`与`j`的下一个元素进行匹配。也就是`s[i]==p[j+1]`
>
> ### 关键点
>
> 1. **求`next`数组：**由于加了哨兵，因此实际上字符串下标从`1`开始。第一个字符最长公共前后缀长度为`0`即`next[1]=0`。求next数组`i`从`2`开始，`j`从`0`开始（因为要与`j+1`进行匹配`p[i]==p[j+1]`）。如果循环结束，当j退回为0，代表此时无法往后退了，j不移动，最长公共前后缀长度为`0`。如果匹配成功，`j`往前移动一位`(j++)`，代表当前以i结尾的最长公共前后缀长度为`j`
> 2. **`kmp`匹配过程：**与求`next`数组类似，不过匹配的是主串与模式串（`s[i]==p[j+1]`）匹配成功`j++`,不成功退回到`j`的最长前后缀长度（`j=next[j]`）。最后如果`j`移动到末尾，也就是`j==m`那么代表匹配成功，第一个匹配成功字符位置为（`i-j`）（如果定义下标从`0`开始，那么为`i-j+1`）

```c++
//带哨兵情况（背诵）
class Solution {
public:
    int strStr(string s, string p) {
        int m=p.size(),n=s.size();
        s.insert(s.begin(),' ');
        p.insert(p.begin(),' ');
        vector<int> next(m+1,0);
        for(int i=2,j=0;i<=m;i++){
            while(j&&p[i]!=p[j+1])j=next[j];
            if(p[i]==p[j+1])j++;
            next[i]=j;
        }
        for(int i=1,j=0;i<=n;i++){
            while(j&&s[i]!=p[j+1])j=next[j];
            if(s[i]==p[j+1])j++;
            if(j==m){
                return i-j;
            }
        }
        return -1;
    }
};


//不带哨兵情况
class Solution {
public:
    int strStr(string s, string p) {
        int m=p.size(),n=s.size();
        vector<int> next(m,0);
        next[0]=-1;
        //j==-1代表无法再回退了。
        for(int i=1,j=-1;i<m;i++){
            while(j!=-1&&p[i]!=p[j+1])j=next[j];
            if(p[i]==p[j+1])j++;
            next[i]=j;
        }

        for(int i=0,j=-1;i<n;i++){
            while(j!=-1&&s[i]!=p[j+1])j=next[j];
            if(s[i]==p[j+1]) j++;
            if(j==m-1){
                return i-m+1;
            }
        }
        return -1;
    }
};


```

## 10.回溯

### 子集型回溯

#### [17. 电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/)

> 利用`DFS`进行遍历，同时在递归的同时要注意状态的恢复，字符串string temp=path+c;并不会影响到原来的字符串path。
>

```c++
class Solution {
public:
    // 深度优先搜索函数
    // u 当前处理的字符位置
    // res 存储最终结果的容器
    // path 当前递归路径的字符串
    // n 输入数字字符串的长度
    // MAPPING 数字到字母的映射表
    // digits 输入的数字字符串
    void dfs(int u, vector<string>& res, string path, int n, string MAPPING[], string& digits) {
        if (u == n) {
            res.emplace_back(path);  // 如果递归到最后一层，将路径加入结果集
            return;
        }
        for (char c : MAPPING[digits[u] - '0']) {  // 遍历当前数字对应的所有字符
            string temp = path + c;  // 将当前字符加入路径
            dfs(u + 1, res, temp, n, MAPPING, digits);  // 递归处理下一个数字
        }
    }

    // 主函数，生成数字字符串的所有字母组合
    vector<string> letterCombinations(string digits) {
        string MAPPING[10] = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};  // 数字到字母的映射
        vector<string> res;
        int n = digits.size();
        if (n == 0) return res;  // 如果输入字符串为空，返回空结果
        dfs(0, res, "", n, MAPPING, digits);  // 开始深度优先搜索
        return res;
    }
};
```
#### [78. 子集](https://leetcode.cn/problems/subsets/)

> 子集回溯，考虑两种思路
>
> 1. 输入角度：选与不选，那么叶子节点为结果
> 2. 答案角度：每次必须选择一个，每一个节点都是答案

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    //当前操作？选或不选当前元素
    //子操作？构造>=i的序列
    //下一个子问题构造>=i+1的序列
    void dfs(int u,int n,vector<int> &nums){
        if(u==n){
            res.emplace_back(path);
            return ;
        }
        //不选
        dfs(u+1,n,nums);

        //选择
        path.emplace_back(nums[u]);
        dfs(u+1,n,nums);
        path.pop_back();
    }
    vector<vector<int>> subsets(vector<int>& nums) {
        dfs(0,nums.size(),nums);
        return res;
    }
};


class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    //当前操作？挑选当前位置u的元素
    //子操作？构造>=i的序列
    //下一个子问题构造>=i+1的序列
    void dfs(int u,int n,vector<int> &nums){
         res.emplace_back(path);
        if(u==n){
           
            return ;
        }
        for(int i=u;i<n;i++){
            path.emplace_back(nums[i]);
            dfs(i+1,n,nums);
            path.pop_back();
        }
    }
    vector<vector<int>> subsets(vector<int>& nums) {
        dfs(0,nums.size(),nums);
        return res;
    }
};
```

#### [257. 二叉树的所有路径](https://leetcode.cn/problems/binary-tree-paths/)

> 本题依然要采用回溯，但是由于是字符串路径，因此我们的`path`，不用设置成全局变量，利用值传递，传递到递归的下一层，下一层的变化不会影响上一层的变化。或者可以使用全局变量，那么就需要显式的进行回溯了

```c++
class Solution {
public:
    vector<string> res; // 存储所有路径的结果向量

    // 深度优先搜索函数
    void dfs(TreeNode *root, string path) {
        if (root == nullptr) // 如果当前节点为空，直接返回
            return;
        
        path += to_string(root->val); // 将当前节点的值添加到路径中
        
        // 如果当前节点是叶子节点（没有左子树和右子树）
        if (root->left == nullptr && root->right == nullptr)
            res.emplace_back(path); // 将完整路径添加到结果中

        // 递归调用左子树和右子树，并在路径中添加"->"
        dfs(root->left, path + "->");
        dfs(root->right, path + "->");
    }

    // 主函数，生成所有从根节点到叶子节点的路径
    vector<string> binaryTreePaths(TreeNode* root) {
        dfs(root, ""); // 从根节点开始深度优先搜索，初始路径为空字符串
        return res;    // 返回所有路径结果
    }
};

```

[113. 路径总和 II](https://leetcode.cn/problems/path-sum-ii/)

> 思路：从根节点开始，每次减去`root->val`，避免了创建额外变量，同时，由于采用的是值传递，隐含了一个回溯，递归下一层的变化，并不会影响上一层。`path`存放路径。同时`path`需要不断的回溯

```c++
class Solution {
public:
    vector<vector<int>> res; // 存储所有满足条件的路径
    vector<int> path;        // 存储当前路径

    // 深度优先搜索函数
    void dfs(TreeNode *root, int targetSum) {
        if (root == nullptr) // 如果当前节点为空，直接返回
            return;

        targetSum -= root->val; // 减去当前节点的值
        path.emplace_back(root->val); // 将当前节点的值添加到路径中

        // 如果当前节点是叶子节点，并且路径和等于目标和
        if (root->left == nullptr && root->right == nullptr && targetSum == 0) {
            res.emplace_back(path); // 将当前路径添加到结果中
        }

        // 递归调用左子树和右子树
        dfs(root->left, targetSum);
        dfs(root->right, targetSum);

        // 回溯，撤销上一步的操作
        path.pop_back();
    }

    // 主函数，查找所有从根节点到叶子节点，路径和等于目标和的路径
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        dfs(root, targetSum); // 从根节点开始深度优先搜索
        return res;           // 返回所有满足条件的路径
    }
};
```

#### [2397. 被列覆盖的最多行数](https://leetcode.cn/problems/maximum-rows-covered-by-columns/)

> 思路：两种思考方式，当前列选与不选，或者，每次必须选择列，
>
> 1. 输入角度，选与不选，需要注意的是，边界条件遍历的列不能超出范围，同时选的列不能超过`numSelect`
> 2. 答案角度，每次必须选择，边界条件为，可供选择的列为0

```c++
class Solution {
public:
    
    //当前操作？该列选还是不选
    //子问题？在i,n中选择一列进行覆盖
    //下一个子问题？在i+1,n中选择一列进行覆盖
    // 当前最大覆盖的行数
    int res = 0;
    // 用一个数组统计所有行中的1的数量
    int cnt[20]{0}; // 假设矩阵的行数不超过20

    // 深度优先搜索函数
    void dfs(int u, int n, vector<vector<int>>& matrix, int numSelect) {
        // 基准条件：当numSelect为0或所有列都已经考虑过
        if (numSelect == 0 || u == matrix[0].size()) {
            // 检查覆盖了多少行
            int temp = 0;
            for (int i = 0; i < n; i++) {
                if (cnt[i] == 0) // 如果该行所有1都被覆盖
                    temp++;
            }
            res = max(res, temp); // 更新最大覆盖行数
            return;
        }

        // 不选择当前列
        dfs(u + 1, n, matrix, numSelect);

        // 选择当前列
        // 将这一列中所有的1所在行的计数器减1
        for (int j = 0; j < n; j++) {
            if (matrix[j][u] == 1)
                cnt[j]--;
        }
        numSelect--; // 减少一个可选列数
        dfs(u + 1, n, matrix, numSelect);

        // 回溯
        // 恢复选择这一列前的状态
        for (int j = 0; j < n; j++) {
            if (matrix[j][u] == 1)
                cnt[j]++;
        }
        numSelect++; // 恢复可选列数
    }

    // 主函数，返回可以覆盖的最大行数
    int maximumRows(vector<vector<int>>& matrix, int numSelect) {
        // 初始化每一行中1的数量
        for (int i = 0; i < matrix.size(); i++) {
            for (int j = 0; j < matrix[0].size(); j++) {
                if (matrix[i][j] == 1)
                    cnt[i]++;
            }
        }
        // 开始深度优先搜索
        dfs(0, matrix.size(), matrix, numSelect);
        return res; // 返回最大覆盖行数
    }
};

```

#### [93. 复原 IP 地址](https://leetcode.cn/problems/restore-ip-addresses/)

> 思路:从答案视角出发，切割字符串，同时思考回溯三问？
>
> 1. 当前操作？切割子串，[u,i]
> 2. 子问题？构造下标>=i的子串
> 3. 下一个子问题，构造下标>=i+1子串
>
> #### 关键点
>
> 1. 剪枝1，如果有前导0，进行剪枝。` if (i != u && s[u] == '0')`，这句话代表的含义是，当前切割的子串为[u,i],如果当前切割不是单字符，那么就必须移除前导0，同时，由于此时`i`一定不等于`u`，可能为`u+1,u+2….`，但如果子串第一个字符`s[u]==‘0’`说明这是一个前导0子串，需要剪枝。
> 2. 剪枝2，如果一部分子串数字大于255说明不符合，需要剪枝

```c++
class Solution {
public:
    vector<string> res; // 用于存储所有有效的IP地址组合
    vector<string> path; // 用于存储当前部分IP地址组合

    // 深度优先搜索函数
    void dfs(int u, int n, string &s) {
        // 如果已经处理到字符串末尾，并且路径中有4部分，说明找到了一个有效的IP地址
        if (u == n && path.size() == 4) {
            string temp = "";
            for (int i = 0; i < 4; i++) {
                if (i != 0)
                    temp += "."; // 在每个部分之间添加'.'
                temp += path[i];
            }
            res.emplace_back(temp); // 将当前IP地址添加到结果中
            return;
        }

        // 遍历从当前位置开始的每一个子串
        for (int i = u; i < n; i++) {
            // 如果当前子串有前导0，则跳过（IP地址的每一部分不能有前导0）
            if (i != u && s[u] == '0')
                return;
            
            long long sum = 0; // 用于计算当前子串转换成的整数值
            // 将子串转换为整数
            for (int j = u; j <= i; j++)
                sum = sum * 10 + s[j] - '0';
            
            // 如果数值超过255，则不是有效的IP地址部分，跳过
            if (sum > 255)
                return;
            
            // 将当前子串添加到路径中
            path.emplace_back(s.substr(u, i - u + 1));
            // 递归处理剩余的字符串
            dfs(i + 1, n, s);
            // 回溯，移除当前子串
            path.pop_back();
        }
    }

    // 主函数，返回所有可能的IP地址组合
    vector<string> restoreIpAddresses(string s) {
        // 从字符串的第一个字符开始进行深度优先搜索
        dfs(0, s.size(), s);
        // 返回所有有效的IP地址
        return res;
    }
};

```



### 组合型回溯

#### [77. 组合](https://leetcode.cn/problems/combinations/)

> 两种思路，根据题目，能够看出，不需要进行去重剪枝
>
> 1. 选与不选
> 2. 必须选择一个数

```c++
//必须选择一个数
//当前操作是什么？选择一个数加入集合中
//子问题是什么，在i,n中选择一个数，加入集合中
//下一个子问题是什么，在i+1,n中选择一个数，加入到集合中。

class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;

    void dfs(int u, int n, int k) {
        if (path.size() == k) { // 如果当前组合长度已经达到k，保存结果并返回
            res.emplace_back(path);
            return;
        }
        
        // 剩余的元素数量不足以构成一个有效的组合时提前返回
        if (path.size() + (n - u + 1) < k) {
            return;
        }
        
        for (int i = u; i <= n; i++) {
            path.emplace_back(i); // 选择当前元素
            dfs(i + 1, n, k); // 递归处理剩余的元素
            path.pop_back(); // 回溯，撤销选择
        }
    }

    vector<vector<int>> combine(int n, int k) {
        dfs(1, n, k); // 从数字1开始进行递归
        return res; // 返回所有组合结果
    }
};

//选与不选

//当前操作？选不选当前元素
//子问题？在u,n中进行询问
//下一个子问题，在u+1,n中进行询问
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;

    void dfs(int u, int n, int k) {
        if (path.size() == k) { // 如果当前组合长度已经达到k，保存结果并返回
            res.emplace_back(path);
            return;
        }
        
        // 剩余的元素数量不足以构成一个有效的组合时提前返回
        if (path.size() + (n - u + 1) < k) {
            return;
        }
        
        dfs(u+1,n,k);

        path.emplace_back(u);
        dfs(u+1,n,k);
        path.pop_back();
    }

    vector<vector<int>> combine(int n, int k) {
        dfs(1, n, k); // 从数字1开始进行递归
        return res; // 返回所有组合结果
    }
};


```

#### [39. 组合总和](https://leetcode.cn/problems/combination-sum/)

> 本题给出条件，通过元素可以重复使用，那么说明，我们递归的时候，选择元素时，可以重复选择，因此，写成`dfs(i)`。
>
> 两种思路
>
> 1. 选与不选
> 2. 每次选择一个
>
> #### 剪枝
>
> 1. 当`target<0`返回剪枝
> 2. 当`u==n`递归到了尽头，返回。



```c++
//选择一个
//本次操作？选择一个数加入集合
//子问题，从i-n中选择一个数，加入集合，
//下一个子问题，从i-n中选择一个数，加入集合(可以重复选择)

class Solution {
public:
    vector<vector<int>> res;  // 用于存储所有满足条件的组合结果
    vector<int> path;         // 用于存储当前组合路径

    // 深度优先搜索函数
    void dfs(int u, int n, vector<int>& candidates, int target) {
        // 如果目标值变为0，表示找到一个满足条件的组合
        if (target == 0) {
            res.emplace_back(path);  // 将当前路径加入结果集
            return;
        }
        // 如果索引达到n或目标值变为负数，表示无效组合
        if (u == n || target < 0)
            return;
        
        // 遍历从索引u到n的所有候选数
        for (int i = u; i < n; i++) {
            path.emplace_back(candidates[i]);  // 选择当前候选数
            dfs(i, n, candidates, target - candidates[i]);  // 递归，继续选择当前数或后续的数
            path.pop_back();  // 回溯，撤销选择
        }
    }

    // 组合总和函数
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        dfs(0, candidates.size(), candidates, target);  // 从索引0开始递归
        return res;  // 返回所有组合结果
    }
};


//选与不选
//本次操作？是否选择该数
//子问题，从i-n中选择一个数，加入集合，
//下一个子问题，从i-n中选择一个数，加入集合(可以重复选择)
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    void dfs(int u,int n,vector<int> &candidates,int target){
        if(target==0){
            res.emplace_back(path);
            return ;
        }
        if(u==n||target<0)
            return ;
        //不选
        dfs(u+1,n,candidates,target);

        //选
        path.emplace_back(candidates[u]);
        dfs(u,n,candidates,target-candidates[u]);
        path.pop_back();
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        dfs(0,candidates.size(),candidates,target);
        return res;
    }
};
```

#### [22. 括号生成](https://leetcode.cn/problems/generate-parentheses/)

> 思路：本题从选与不选的角度入手比较方便，假设选择左括号为不选，选择右括号为选。根据题目意思，那么明显必须`left<n`才能选左括号，如果`right<left`才能添加右括号

```c++
class Solution {
public:
    vector<string> res;
    string path;
    int left=0,right=0;
    void dfs(int u,int n){
        if(left==n&&right==n){
            res.emplace_back(path);
            return ;
        }
        if(left<n){
            path.push_back('(');
            left++;
            dfs(u+1,n);
            path.pop_back();
            left--;
        }
        if(right<left){
            right++;
            path.push_back(')');
            dfs(u+1,n);
            path.pop_back();
            right--;
        }
    }        
    vector<string> generateParenthesis(int n) {
        path="";
        dfs(0,n);
        return res;
    }
};
```

#### [40. 组合总和 II](https://leetcode.cn/problems/combination-sum-ii/)

> 思路：本题的重点在于，每个数字仅能使用一次，这意味着，我们在循环中，应该写成`dfs(i+1)`。同时注意解集中不能包括重复元素。这意味着，我们需要对同一层（`for`循环）的树层进行剪枝处理。
>
> #### 关键点：
>
> **去重：**去重的前提是要先对数组进行一个排序，这样保证了相同元素挨在一起出现，同时，如果出现相同元素，那么第一个元素能进行搜索的范围永远是最大的，他会将同一层相同元素能够搜索出来的答案包含在内。那么我们仅需要保留第一条分支即可，其他同层相同元素进行剪枝处理。

```c++
class Solution {
public:
    vector<vector<int>> res;  // 存储所有符合条件的组合结果
    vector<int> path;         // 当前递归路径中的候选数
    vector<bool> st;          // 标记数组，用于记录元素是否被使用过

    void dfs(int u, int n, int target, vector<int>& candidates) {
        if (target == 0) {  // 如果目标值为0，表示找到了一个满足条件的组合
            res.emplace_back(path);  // 将当前路径加入结果集
            return;
        }
        if (u == n || target < 0 || candidates[u] > target) {
            return;  // 递归终止条件：索引超出范围、目标值为负数或当前元素大于目标值时返回
        }
        
        for (int i = u; i < n; i++) {
            if (i > u && candidates[i] == candidates[i - 1]) {
                continue;  // 去重操作：跳过相邻重复的元素，确保在同一层级不重复使用相同的元素
            }
            
            path.emplace_back(candidates[i]);  // 选择当前元素
            dfs(i + 1, n, target - candidates[i], candidates);  // 递归调用，更新目标值并继续选择下一个元素
            path.pop_back();  // 回溯操作：撤销选择的当前元素，尝试下一个元素
        }
    }

    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        st.resize(candidates.size(), false);  // 初始化标记数组，大小与候选数组相同，默认为false
        sort(candidates.begin(), candidates.end());  // 对候选数组进行排序，便于后续去重操作
        dfs(0, candidates.size(), target, candidates);  // 调用深度优先搜索函数进行组合求解
        return res;  // 返回所有符合条件的组合结果
    }
};

```

#### [216. 组合总和 III](https://leetcode.cn/problems/combination-sum-iii/)

> 两种思路，选与不选，每次必须选择一个数

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    void dfs(int u,int n,int k){
        if(path.size()==k&&n==0){
            res.emplace_back(path);
            return ;
        }
        if(n<0||path.size()>k||((u+9)*(9-u+1))/2<n)
            return  ;
        dfs(u+1,n,k);

        path.emplace_back(u);
        dfs(u+1,n-u,k);
        path.pop_back();
    }
    vector<vector<int>> combinationSum3(int k, int n) {
        dfs(1,n,k);
        return res;
    }
};

class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    void dfs(int u,int n,int k){
        if(path.size()==k&&n==0){
            res.emplace_back(path);
            return ;
        }
        if(n<0||path.size()>k)
            return  ;
        for(int i=u;i<=9;i++){
            path.emplace_back(i);
            dfs(i+1,n-i,k);
            path.pop_back();
        }
    }
    vector<vector<int>> combinationSum3(int k, int n) {
        dfs(1,n,k);
        return res;
    }
};
```

### 排列型回溯

#### [46. 全排列](https://leetcode.cn/problems/permutations/)

> ### 关键点
>
> 1. 需要使用一个`bool`类型数组`st`记录已经加入了`path`的数，
> 2. 每次遍历需要从`0`开始，因为全排列中每个元素在每个位置上都可能出现

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    vector<bool> st;//标记已经加入了path中的元素
    //当前操作？选择path[u]的值
    //子问题？构造path中>=u的序列
    //下一个子问题?构造path中>=i+1的序列
    void dfs(int u,int n,vector<int> &nums){
        if(u==n){
            res.emplace_back(path);
            return ;
        }
        for(int i=0;i<n;i++){
            if(st[i])
                continue;
            //标记当前nums[i]已经加入了集合中。
            st[i]=true;
            path.emplace_back(nums[i]);
            dfs(u+1,n,nums);
            //回溯
            st[i]=false;
            path.pop_back();
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        st.resize(nums.size(),false);
        dfs(0,nums.size(),nums);
        return res;
    }
};
```

#### [47. 全排列 II](https://leetcode.cn/problems/permutations-ii/)

> 思路跟全排列一致，不过需要进行同一树层剪枝，`if(i!=0&&nums[i]==nums[i-1]&&!st[i-1])`

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    vector<bool> st;
    void dfs(int u,int n,vector<int> &nums){
        if(u==n){
            res.emplace_back(path);
            return ;
        }
        for(int i=0;i<n;i++){
            if(i!=0&&nums[i]==nums[i-1]&&!st[i-1])
                continue;
            if(st[i])
                continue;
            st[i]=true;
            path.emplace_back(nums[i]);
            dfs(u+1,n,nums);
            path.pop_back();
            st[i]=false;
        }
    }
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        sort(nums.begin(),nums.end());
        st.resize(nums.size(),false);
        dfs(0,nums.size(),nums);
        return res;
    }
};
```

#### [51. N 皇后](https://leetcode.cn/problems/n-queens/)

> 也是一个全排列的经典问题，主要思路就是在棋盘上放置n个皇后，要保证行列对角反对角，都没有皇后才能放置。可以从两种角度思考
>
> 1. 放与不放（从第一个位置开始往右下角搜索，行列对角反对角都必须记录）
> 2. 每次必须放置一个。（从第一行开始往下搜寻，可以少写一个行的记录数组）

```c++
class Solution {
public:
    vector<vector<string>> res; // 存储所有可能的N皇后解法
    vector<string> path; // 临时路径存储单个解法
    vector<bool> row, col, dg, udg; // 标记行、列和对角线是否被占用
    char g[11][11]; // 棋盘

    // dfs(x, y, n, count) 代表决定当前是否要在位置 (x, y) 放置皇后
    void dfs(int x, int y, int n, int count) {
        // 如果列数超过 n，换到下一行的第一列
        if (y == n) {
            x++;
            y = 0;
        }
        
        // 如果行数达到 n，检查是否放置了 n 个皇后
        if (x == n) {
            if (count != n)
                return;
            vector<string> currentSolution;
            for (int i = 0; i < n; i++) {
                currentSolution.emplace_back(g[i]);
            }
            res.emplace_back(currentSolution);
            return;
        }

        // 如果当前行数大于剩余需要放置的皇后数，剪枝返回
        if (x > count) {
            return;
        }

        // 当前位置不放置皇后
        dfs(x, y + 1, n, count);

        // 当前位置放置皇后
        if (!row[x] && !col[y] && !dg[x + y] && !udg[y - x + n]) {
            row[x] = col[y] = dg[x + y] = udg[y - x + n] = true;
            g[x][y] = 'Q';
            dfs(x, y + 1, n, count + 1);
            row[x] = col[y] = dg[x + y] = udg[y - x + n] = false;
            g[x][y] = '.';
        }
    }

    // 主函数，初始化并调用 dfs
    vector<vector<string>> solveNQueens(int n) {
        col.resize(n, false);
        dg.resize(2 * n, false);
        udg.resize(2 * n, false);
        row.resize(n, false);

        // 初始化棋盘为空
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                g[i][j] = '.';

        // 从 (0, 0) 开始 dfs
        dfs(0, 0, n, 0);
        return res;
    }
};


//每次必须选择一个
class Solution {
public:
    vector<vector<string>> res; // 存储所有可能的N皇后解法
    vector<string> path; // 存储当前解法的棋盘状态
    vector<bool> col, dg, udg; // 标记列、主对角线、副对角线是否被占用
    char g[10][10]; // 棋盘

    // dfs函数，x代表当前处理的行，n代表棋盘的大小
    void dfs(int x, int n) {
        // 如果x等于n，表示已经放置完最后一行
        if (x == n) {
            for (int i = 0; i < n; i++)
                path.emplace_back(g[i]); // 将当前棋盘状态加入path
            res.emplace_back(path); // 将当前解法加入结果集
            path.clear(); // 清空path，为下一次解法做准备
            return;
        }

        // 尝试在当前行的每一列放置皇后
        for (int i = 0; i < n; i++) {
            // 检查当前列、主对角线、副对角线是否被占用
            if (!col[i] && !dg[i + x] && !udg[i - x + n]) {
                col[i] = dg[i + x] = udg[i - x + n] = true; // 标记当前列、主对角线、副对角线被占用
                g[x][i] = 'Q'; // 在棋盘上放置皇后
                dfs(x + 1, n); // 递归处理下一行
                col[i] = dg[i + x] = udg[i - x + n] = false; // 回溯，取消当前列、主对角线、副对角线的占用
                g[x][i] = '.'; // 移除棋盘上的皇后
            }
        }
    }
    // 解决N皇后问题的主函数
    vector<vector<string>> solveNQueens(int n) {
        col.resize(n, false); // 初始化列标记
        dg.resize(2 * n, false); // 初始化主对角线标记
        udg.resize(2 * n, false); // 初始化副对角线标记

        // 初始化棋盘为空
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                g[i][j] = '.';

        dfs(0, n); // 从第0行开始递归搜索解法
        return res; // 返回所有解法
    }
};

```



### 字符串回溯

#### [131. 分割回文串](https://leetcode.cn/problems/palindrome-partitioning/)

> 思路：不断的切割枚举子串，检查当前切割的子串是否符合回文串定义，如果符合，那么直接加入集合中即可，不符合，跳过当前，进入同一树层的下一个分支。

```c++
class Solution {
public:
    vector<vector<string>> res;
    vector<string> path;
    bool isprime(string &s,int left,int right){
        while(left<right){
            if(s[left]!=s[right])
                return false;
            left++;
            right--;
        }
        return true;
    }
    void dfs(int u,int n,string &s){
        if(u==n){
            res.emplace_back(path);
            return ;
        }
        //当前探索的位置区域为u,n-1
        for(int i=u;i<n;i++){
            string temp=s.substr(u,i-u+1);
            //截取字符串长度应为i-u+1;
            if(isprime(s,u,i)){
                path.emplace_back(temp);
                dfs(i+1,n,s);
                path.pop_back();
            }
        }
    }
    vector<vector<string>> partition(string s) {
        dfs(0,s.size(),s);
        return res;
    }
};
```



## 11.图论

### 拓扑排序

#### [207. 课程表](https://leetcode.cn/problems/course-schedule/)

> 思路：采用`BFS`进行搜索，将入度为0的节点加入到队列中，每次将当前节点的单链表上节点入度减一，如果减为0，加入到队列中，最后检查加入过队列的节点个数是否为图中节点个数。
>
> ### 拓扑排序的基本思路
>
> 1. **初始化**：
>    - 构建邻接表来表示图。
>    - 计算每个节点的入度`d[N]`表示。
> 2. **寻找入度为0的节点**：
>    - 将所有入度为0的节点加入队列。这些节点没有任何前驱，可以作为排序的起点。
> 3. **处理队列**：
>    - 从队列中取出一个节点，加入排序结果。
>    - 遍历该节点的所有邻接节点，将它们的入度减1。如果某个邻接节点的入度减为0，则将其加入队列。
>    - 重复以上步骤，直到队列为空。
> 4. **检查结果**：
>    - 如果排序结果中的节点数等于图中的节点数，则拓扑排序成功。否则，图中存在环，无法进行拓扑排序。
>
> ### DFS递归过程
>
> 定义一个`int`类型的`flag`数组，数组中每个节点有三种状态分别是0,1,-1,`0`表示当前节点未被访问，`1`表示当前节点正在被访问。`-1`表示当前节点访问完毕。（原理，遍历图中每个节点的每条路径，如果在递归过程中发现一个顶点被在同一个起点出发的DFS中被访问了两次，说明有环。）
>
> 递归过程：
>
> 1. 借助一个标志列表 flags，用于判断每个节点 i （课程）的状态：
>    1. 未被 DFS 访问：i == 0；
>    2. 已被其他节点启动的 DFS 访问：i == -1；
>    3. 已被当前节点启动的 DFS 访问：i == 1。
>
> 2. 对 `numCourses` 个节点依次执行 DFS，判断每个节点起步 DFS 是否存在环，若存在环直接返回 false。DFS 流程；
>    1. 终止条件：
>       1. 当 flag[i] == -1，说明当前访问节点已被其他节点启动的 DFS 访问，无需再重复搜索，直接返回 True。
>       2. 当 flag[i] == 1，说明在本轮 DFS 搜索中节点 i 被第 2 次访问，即 课程安排图有环 ，直接返回 False。
>    2. 将当前访问节点 i 对应 flag[i] 置 1，即标记其被本轮 DFS 访问过；
>    3. 递归访问当前节点 i 的所有邻接节点 j，当发现环直接返回 False；
>    4. 当前节点所有邻接节点已被遍历，并没有发现环，则将当前节点 flag 置为 −1 并返回 True。
> 3. 若整个图 DFS 结束并未发现环，返回 `true`

```c++
class Solution {
public:
    // 常量N，用于定义最大节点数
    static const int N = 4000;
    int h[N], e[N], ne[N], idx; // 邻接表的数组和索引
    int d[N]; // 存储每个点的入度

    // 构造函数，初始化邻接表头指针和索引
    Solution() {
        memset(h, -1, sizeof h);
        idx = 0;
    }

    // 添加边的方法，从节点a到节点b
    void add(int a, int b) {
        e[idx] = b;       // e数组存储边的目标节点
        ne[idx] = h[a];   // ne数组存储当前边的下一个边的索引
        h[a] = idx++;     // h数组存储某个节点的第一条边的索引
    }

    // 拓扑排序方法，判断是否可以完成所有课程
    bool topsort(int numCourses) {
        int count = 0; // 记录已经处理节点个数
        queue<int> q;  // 存储入度为0的节点

        // 所有入度为0的节点入队
        for (int i = 0; i < numCourses; i++) {
            if (d[i] == 0)
                q.push(i);
        }

        // 处理队列中的节点
        while (!q.empty()) {
            int t = q.front(); // 取出队首节点
            q.pop();
            count++;
            // 遍历当前节点的所有邻接节点
            for (int i = h[t]; i != -1; i = ne[i]) {
                int j = e[i];  // 获取邻接节点
                if (--d[j] == 0) { // 邻接节点的入度减1，如果减为0，则入队
                    q.push(j);
                }
            }
        }

        // 如果处理的节点数等于课程总数，则可以完成所有课程，否则不能
        return count == numCourses;
    }

    // 主方法，判断是否可以完成所有课程
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        memset(h, -1, sizeof h); // 初始化邻接表头指针
        memset(d, 0, sizeof d);  // 初始化入度数组

        // 根据先修课程关系构建图
        for (int i = 0; i < prerequisites.size(); i++) {
            add(prerequisites[i][1], prerequisites[i][0]); // 添加边
            d[prerequisites[i][0]]++; // 更新入度
        }

        // 调用拓扑排序方法判断是否可以完成所有课程
        return topsort(numCourses);
    }
};

//DFS判断是否有环
class Solution {
public:
    static const int N=2500, M=6000;  // N表示最大课程数，M表示最大边数（先修课程关系数）
    int h[N], e[M], ne[M], idx;       // h[]: 邻接表头结点，e[]: 边的目标节点，ne[]: 下一个边的索引，idx: 边的索引
    int flag[N];                      // 记录每个顶点的状态，0表示未被访问，1表示正在访问，-1表示访问完毕
    
    // 添加边 a -> b
    void add(int a, int b) {
        e[idx] = b;
        ne[idx] = h[a];
        h[a] = idx++;
    }

    // 深度优先搜索函数
    bool dfs(int n, int x) {
        if (flag[x] == 1) return false;   // 如果当前节点正在被访问，说明存在环，返回false
        if (flag[x] == -1) return true;   // 如果当前节点已访问完毕，直接返回true

        flag[x] = 1;  // 标记当前节点为正在访问
        // 遍历邻接表
        for (int i = h[x]; i != -1; i = ne[i]) {
            int j = e[i];
            if (!dfs(n, j)) return false;  // 递归访问邻接节点，如果存在环，返回false
        }
        flag[x] = -1;  // 标记当前节点访问完毕
        return true;
    }

    // 判断是否可以完成所有课程
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        memset(h, -1, sizeof h);   // 初始化邻接表头结点
        memset(flag, 0, sizeof flag);  // 初始化每个顶点的访问状态
        idx = 0;  // 初始化边的索引

        // 构建图的邻接表
        for (int i = 0; i < prerequisites.size(); i++) {
            int a = prerequisites[i][0], b = prerequisites[i][1];
            add(b, a);  // 先修课程关系 b -> a
        }

        // 遍历每个课程，对未访问的课程进行DFS
        for (int i = 0; i < numCourses; i++) {
            if (!dfs(numCourses, i)) return false;  // 如果检测到环，返回false
        }

        return true;  // 如果没有检测到环，返回true
    }
};

```

### 最短路径

#### [1091. 二进制矩阵中的最短路径](https://leetcode.cn/problems/shortest-path-in-binary-matrix/)

> 思路：题目要求最短路径，在网格中走一步距离为1，因此可以使用`BFS`进行求，每次处理一层节点，判断能否到达。如果能到达就加入队列中，将其标记为已经访问过。

```c++
#include <vector>
#include <queue>
#include <iostream>

using namespace std;

class Solution {
public:
    int shortestPathBinaryMatrix(vector<vector<int>>& grid) {
        int n = grid.size();
        if (grid[0][0] == 1 || grid[n-1][n-1] == 1) return -1; // 起点或终点被阻塞
        
        queue<pair<int,int>> q;
        q.push({0, 0});
        grid[0][0] = 1; // 标记起点为已访问
        int res = 0;
        int nx[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
        int ny[8] = {0, 0, -1, 1, -1, 1, -1, 1};
        
        while (!q.empty()) {
            int size = q.size();
            res++; // 增加路径长度
            for (int i = 0; i < size; i++) {
                auto t = q.front();
                int x = t.first, y = t.second;
                q.pop();
                if (x == n - 1 && y == n - 1) return res; // 到达终点
                for (int j = 0; j < 8; j++) {
                    int tx = x + nx[j], ty = y + ny[j];
                    if (tx >= 0 && tx < n && ty >= 0 && ty < n && grid[tx][ty] == 0) {
                        grid[tx][ty] = 1; // 标记为已访问
                        q.push({tx, ty});
                    }       
                }
            }
        }
        return -1; // 无法到达终点
    }
};

//判断是否有环

```

#### [743. 网络延迟时间](https://leetcode.cn/problems/network-delay-time/)

> 利用最短路径算法计算到顶点的最长路径（5种均可以使用），题目给的是一个稠密图，因此使用朴素dijkstra比较合适。同时要注意下标，题目下标从1开始。
>
> ### 关键点
>
> 1. 题目所给为稠密图
> 2. 节点下标从1开始。
> 3. 所给图是有向图

```c++
class Solution {
public:
    //朴素dijkstra，邻接矩阵存储
    static const int N=110,INF=0x3f3f3f3f;
    int g[N][N];
    int dist[N];
    bool st[N];
    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
        memset(dist,0x3f,sizeof dist);
        memset(g,0x3f,sizeof g);
        //初始化邻接矩阵
        for(int i=0;i<times.size();i++){
            int a=times[i][0],b=times[i][1],w=times[i][2];
            g[a][b]=w;
        }
        dist[k]=0;
        for(int i=0;i<n;i++){
            int t=-1;
            for(int j=1;j<=n;j++){
                if(!st[j]&&(t==-1||dist[j]<dist[t]))
                    t=j;
            }
            st[t]=true;

            for(int j=1;j<=n;j++)
                dist[j]=min(dist[j],dist[t]+g[t][j]);
        }
        int ans=0;
        for(int i=1;i<=n;i++){
            if(dist[i]>=0x3f3f3f3f/2)return -1;
            ans=max(ans,dist[i]);
        }
        return ans;
    }
};
```

#### [2642. 设计可以求最短路径的图类](https://leetcode.cn/problems/design-graph-with-shortest-path-calculator/)

> 利用最短路径算法求解。题目没有负环优先使用`spfa`
>
> ### 关键点
>
> 1. 所给图为稠密图
> 2. 所给图为有向图
> 3. 节点下标从0开始
> 4. 没有自环和重边

```c++
class Graph {
public:
    static const int N=110,M=10500,INF=0x3f3f3f3f;
    int dist[N];
    int g[N][N];
    bool st[N];
    int n;
    // void add(int a,int b,int c){
    //     e[idx]=b;
    //     w[idx]=c;
    //     ne[idx]=h[a];
    //     h[a]=idx++;
    // }
    Graph(int n, vector<vector<int>>& edges) {
        this->n=n;
        memset(dist,0x3f,sizeof dist);
        //memset(h,-1,sizeof h);
        memset(g,0x3f,sizeof g);
        // for(int i=0;i<=n;i++)
        //     for(int j=0;j<=n;j++){
        //         if(i==j)g[i][j]=0;
        //         else g[i][j]=INF;
        //     }
        for(int i=0;i<edges.size();i++){
            int a=edges[i][0],b=edges[i][1],c=edges[i][2];
            g[a][b]=c;
        }
    }
    
    void addEdge(vector<int> edge) {
        int a=edge[0],b=edge[1],c=edge[2];
        g[a][b]=c;
    }
    
    int shortestPath(int node1, int node2) {
        memset(dist, 0x3f, sizeof dist);  // Reset dist array
        memset(st, false, sizeof st);  
        dist[node1]=0;
        for(int i=1;i<=n;i++){
            int t=-1;
            for(int j=0;j<=n;j++)
                if(!st[j]&&(t==-1||dist[j]<dist[t]))
                    t=j;
            st[t]=true;

            for(int j=0;j<=n;j++)
                dist[j]=min(dist[j],dist[t]+g[t][j]);
        }
        if(dist[node2]>=0x3f3f3f3f/2)return -1;
        return dist[node2];
    }
};

/**
 * Your Graph object will be instantiated and called as such:
 * Graph* obj = new Graph(n, edges);
 * obj->addEdge(edge);
 * int param_2 = obj->shortestPath(node1,node2);
 */


class Graph {
public:
    static const int N = 110, M = 10500;
    int dist[N]; // 数组，用于记录从起点到各点的最短距离
    int h[N], w[M], e[M], ne[M], idx; // 邻接表相关数组
    bool st[N]; // 数组，标记某个节点是否在队列中

    // 添加边的函数
    void add(int a, int b, int c) {
        e[idx] = b; // 记录边的终点
        w[idx] = c; // 记录边的权重
        ne[idx] = h[a]; // 将边加入邻接表
        h[a] = idx++; // 更新邻接表头指针，并自增 idx
    }

    // 构造函数，用于初始化图
    Graph(int n, vector<vector<int>>& edges) {
        memset(dist, 0x3f, sizeof dist); // 将 dist 数组初始化为正无穷
        memset(h, -1, sizeof h); // 将邻接表头指针初始化为 -1
        for (int i = 0; i < edges.size(); i++) {
            int a = edges[i][0], b = edges[i][1], c = edges[i][2];
            add(a, b, c); // 将所有边加入图中
        }
    }

    // 添加单条边的函数
    void addEdge(vector<int> edge) {
        int a = edge[0], b = edge[1], c = edge[2];
        add(a, b, c); // 将边加入图中
    }

    // 求从 node1 到 node2 的最短路径
    int shortestPath(int node1, int node2) {
        memset(dist, 0x3f, sizeof dist); // 重置 dist 数组
        memset(st, 0, sizeof st); // 重置标记数组
        queue<int> q; // 队列，用于存放待处理的节点
        dist[node1] = 0; // 起点到自己的距离为 0
        q.push(node1); // 将起点加入队列
        st[node1] = true; // 标记起点已在队列中

        while (!q.empty()) {
            int t = q.front(); // 取出队首元素
            q.pop(); // 弹出队首元素
            st[t] = false; // 取消标记

            // 遍历 t 的所有邻边
            for (int i = h[t]; i != -1; i = ne[i]) {
                int j = e[i]; // 获取邻边的终点
                if (dist[j] > dist[t] + w[i]) { // 如果找到更短路径
                    dist[j] = dist[t] + w[i]; // 更新最短距离
                    if (!st[j]) { // 如果终点不在队列中
                        q.push(j); // 将终点加入队列
                        st[j] = true; // 标记终点已在队列中
                    }
                }
            }
        }

        // 返回结果，如果无法到达，返回 -1
        if (dist[node2] >= 0x3f3f3f3f / 2) return -1;
        return dist[node2];
    }
};

```

#### [1514. 概率最大的路径](https://leetcode.cn/problems/path-with-maximum-probability/)

> 思路：利用最短路径算法进行计算，权值为概率。我们要寻找一个概率最大的路径，也就是要寻找一条从起点到终点使得概率成绩最大的路径。
>
> ### 转换为概率
>
> 1. 使用Dijkstra算法的思想，但在判断最短路径时，使用概率乘积的比较。
> 2. 初始化起点的概率为1（`dist[start_node] = 1.0`）。
> 3. 在每次选择当前未被处理的最大概率的节点。
> 4. 对于每个相邻节点，尝试用当前节点的概率乘上边的成功概率来更新相邻节点的概率。
> 5. 如果通过当前节点到相邻节点的概率更大，则更新相邻节点的概率。
>
> ### 关键点
>
> 1. 所给图为无向图，添加边时需要添加两次
> 2. 节点下标从0开始
> 3. 所给图为稀疏图（采用`spfa`或者堆优化的`dijkstra`）

```c++
//spfa  时间O(m)最坏O(n*m)
class Solution {
public:
    static const int N = 1e4 + 100, M = 1e5 + 500;  // 定义常量，N 为节点数，M 为边数
    double dist[N];  // 存储从起点到各节点的最大概率
    int h[N], e[M], ne[M], idx;  // 邻接表：h 为头节点数组，e 为边数组，ne 为下一条边的数组，idx 为边的索引
    double w[M];  // 存储每条边的成功概率
    bool st[N];  // 标记某个节点是否已被处理过

    // 添加边的函数
    void add(int a, int b, double c) {
        e[idx] = b;  // 记录边的目标节点
        w[idx] = c;  // 记录边的成功概率
        ne[idx] = h[a];  // 更新邻接表
        h[a] = idx++;  // 更新头节点数组，并将边的索引加一
    }

    // 求从 start_node 到 end_node 的最大概率路径
    double maxProbability(int n, vector<vector<int>>& edges, vector<double>& succProb, int start_node, int end_node) {
        memset(h, -1, sizeof h);  // 初始化头节点数组，-1 表示该节点没有边
        memset(dist, 0, sizeof dist);  // 初始化距离数组，所有节点的最大概率初始为 0
        memset(st, false, sizeof st);  // 初始化标记数组，所有节点初始为未处理状态

        // 添加边
        for (int i = 0; i < edges.size(); i++) {
            add(edges[i][0], edges[i][1], succProb[i]);  // 添加单向边
            add(edges[i][1], edges[i][0], succProb[i]);  // 添加反向边
        }

        queue<int> q;  // 定义队列，用于 SPFA 算法
        q.push(start_node);  // 将起点加入队列
        dist[start_node] = 1.0;  // 起点到自身的概率为 1
        st[start_node] = true;  // 标记起点为已处理状态

        // SPFA 算法主循环
        while (!q.empty()) {
            int t = q.front();  // 取出队首节点
            q.pop();  // 弹出队首节点
            st[t] = false;  // 标记该节点为未处理状态

            // 遍历该节点的所有邻边
            for (int i = h[t]; i != -1; i = ne[i]) {
                int j = e[i];  // 获取邻边的目标节点
                // 如果通过 t 节点到 j 节点的概率更大，则更新 j 节点的最大概率
                if (dist[j] < dist[t] * w[i]) {
                    dist[j] = dist[t] * w[i];
                    // 如果 j 节点未在队列中，则将其加入队列并标记为已处理状态
                    if (!st[j]) {
                        q.push(j);
                        st[j] = true;
                    }
                }
            }
        }
        // 返回终点的最大概率，如果未更新过则为 0
        return dist[end_node];
    }
};



//堆优化dijkstra  O(m*logn)
class Solution {
public:
    typedef pair<double, int> PDI;  // 定义一个类型别名，用于存储概率和节点的对
    static const int N = 1e4 + 100, M = 1e5 + 500;  // 定义常量，N 为节点数，M 为边数
    double dist[N];  // 存储从起点到各节点的最大概率
    int h[N], e[M], ne[M], idx;  // 邻接表：h 为头节点数组，e 为边数组，ne 为下一条边的数组，idx 为边的索引
    double w[M];  // 存储每条边的成功概率
    bool st[N];  // 标记某个节点是否已被处理过

    // 添加边的函数
    void add(int a, int b, double c) {
        e[idx] = b;  // 记录边的目标节点
        w[idx] = c;  // 记录边的成功概率
        ne[idx] = h[a];  // 更新邻接表
        h[a] = idx++;  // 更新头节点数组，并将边的索引加一
    }

    // 求从 start_node 到 end_node 的最大概率路径
    double maxProbability(int n, vector<vector<int>>& edges, vector<double>& succProb, int start_node, int end_node) {
        memset(h, -1, sizeof h);  // 初始化头节点数组，-1 表示该节点没有边
        memset(dist, 0, sizeof dist);  // 初始化距离数组，所有节点的最大概率初始为 0
        memset(st, false, sizeof st);  // 初始化标记数组，所有节点初始为未处理状态

        // 添加边
        for (int i = 0; i < edges.size(); i++) {
            add(edges[i][0], edges[i][1], succProb[i]);  // 添加单向边
            add(edges[i][1], edges[i][0], succProb[i]);  // 添加反向边
        }

        // 定义优先队列（最大堆），按概率从大到小排序
        priority_queue<PDI, vector<PDI>> heap;
        heap.push({1.0, start_node});  // 将起点加入队列，初始概率为 1
        dist[start_node] = 1.0;  // 起点到自身的概率为 1

        // Dijkstra 算法主循环
        while (!heap.empty()) {
            auto t = heap.top();  // 取出队首元素
            heap.pop();  // 弹出队首元素
            int var = t.second;  // 当前节点
            double distance = t.first;  // 当前节点的概率
            if (st[var]) continue;  // 如果该节点已被处理过，跳过
            st[var] = true;  // 标记该节点为已处理状态

            // 遍历该节点的所有邻边
            for (int i = h[var]; i != -1; i = ne[i]) {
                int j = e[i];  // 获取邻边的目标节点
                // 如果通过 var 节点到 j 节点的概率更大，则更新 j 节点的最大概率
                if (dist[j] < distance * w[i]) {
                    dist[j] = distance * w[i];
                    heap.push({dist[j], j});  // 将更新后的节点加入优先队列
                }
            }
        }
        // 返回终点的最大概率，如果未更新过则为 0
        return dist[end_node];
    }
};

```

#### [1631. 最小体力消耗路径](https://leetcode.cn/problems/path-with-minimum-effort/)

> 思路：根据题意，要求找到所有到达右下角路径中，体力值最小的一个，那么，我们可以将其抽象为一个图，图中的节点编号为`(i*n+j)`其中n为列。由于是无向图，因此我们添加的时候需要添加两次。对于dist数组，我们定义为到达k路径中的最小体力值。
>
> 取到达前一个节点的最小体力值，以及指向k的边权值中的最大值，即为新的new_dist[j].如果新的体力值小于dist[j]，那么则更新。
>
> ### 步骤
>
> 1. **图的节点编号**：
>
>    - 将二维矩阵的每个元素视为图的一个节点，节点的编号为 `(i * n + j)`，其中 `i` 是行索引，`j` 是列索引，`n` 是列数。
>
>    - 这种编号方式确保了二维矩阵的每个元素都有唯一的编号。
>
> 2. **无向图的构建**：
>
>    - 由于题目要求无向图，因此每条边都需要添加两次。也就是说，对于每对相邻节点 `(a, b)` 和 `(b, a)` 都要添加边。
>
>    - 边的权值为相邻节点高度差的绝对值，即 `abs(heights[i][j] - heights[x][y])`。
>
> 3. **dist数组的定义**：
>
>    - `dist` 数组用于存储从起点到每个节点的最小体力值。
>
>    - 初始情况下，`dist` 数组的值设置为 `INF`，起点 `dist[0]` 设置为 `0`，表示起点到自己的体力值为 `0`。
>
> 4. **更新最小体力值**：
>
>    - 使用最小堆（优先队列）来维护当前节点的最小体力值，保证每次处理的是当前路径体力值最小的节点。
>
>    - 从堆中取出当前最小体力值的节点 `t`，遍历其邻接节点 `j`，计算新的体力值 `new_dist = max(dist[t], w[i])`，其中 `w[i]` 是边的权值。
>
>    - 如果 `new_dist` 小于 `dist[j]`，则更新 `dist[j]` 并将其加入堆中。
>
> ### 关键点
>
> 1. 图为无向图
> 2. `dist`定义为到达k路径的最小体力值
> 3. 将坐标转化为节点编号
> 4. 稀疏图（堆优化dijkstra或者）

```c++
class Solution {
public:
    typedef pair<int,int> PII;
    static const int N=1e4+100,M=1e5,INF=0x3f3f3f3f;
    int dist[N];//dist表示到达k路径的最大值。
    int h[N],w[M],e[M],ne[M],idx;
    bool st[N];
    int dx[4]={-1,1,0,0},dy[4]={0,0,-1,1};
    void add(int a,int b,int c){
        e[idx]=b;
        w[idx]=c;
        ne[idx]=h[a];
        h[a]=idx++;
    }
    int minimumEffortPath(vector<vector<int>>& heights) {
        int rows=heights.size();
        int col=heights[0].size();
        int count=rows*col;
        memset(dist,0x3f,sizeof dist);
        memset(h,-1,sizeof h);
        //建图
        for(int i=0;i<rows;i++)
            for(int j=0;j<col;j++){
                int a=i*col+j;
                for(int k=0;k<4;k++){
                    int x=(i+dx[k]);
                    int y=j+dy[k];
                    int b=x*col+y;
                    if(x>=0&&x<rows&&y>=0&&y<col){
                        add(a,b,abs(heights[i][j]-heights[x][y]));
                        add(b,a,abs(heights[i][j]-heights[x][y]));
                    }
                }
            }
       priority_queue<PII,vector<PII>,greater<PII>> heap;
       heap.push({0,0});
       dist[0]=0;
       while(heap.size()){
        auto t=heap.top();
        heap.pop();
        int var=t.second,min_distance=t.first;
        if(st[var])continue;
        st[var]=true;
        for(int i=h[var];i!=-1;i=ne[i]){
            int j=e[i];
            if(max(min_distance,w[i])<dist[j]){
                dist[j]=max(min_distance,w[i]);
                heap.push({dist[j],j});
            }
        }
       }
    return dist[rows * col - 1];
    }
};
```





### DFS/BFS

#### 岛屿问题

> 解决岛屿问题思路：
>
> 1. 封闭岛：将边界的岛屿进行标记，之后没有被标记岛屿个数即可
> 2. 

##### [200. 岛屿数量](https://leetcode.cn/problems/number-of-islands/)

> 思路：利用深度优先遍历，不断的向四个方向进行扩展，如果发现当前不是水域，标记当前陆地已被访问。
>
> #### 技巧
>
> 1. `int dx[4] = {-1, 1, 0, 0}, dy[4] = {0, 0, -1, 1};`定义上下左右四个方向的变量，不用写4个dfs

```c++
class Solution {
public:
    // 定义四个方向的向量，分别代表上下左右
    int dx[4] = {-1, 1, 0, 0}, dy[4] = {0, 0, -1, 1};
    int res = 0; // 结果变量，记录岛屿的数量

    // 深度优先搜索函数，用于遍历和标记岛屿
    void dfs(int x, int y, vector<vector<char>>& grid) {
        // 如果当前位置在矩阵范围内并且为'1'（即未访问的陆地）
        if (x >= 0 && x < grid.size() && y >= 0 && y < grid[0].size() && grid[x][y] == '1') {
            // 标记为已访问（使用字符'2'表示）
            grid[x][y] = '2';
            // 遍历四个方向的相邻节点
            for (int i = 0; i < 4; i++) {
                dfs(x + dx[i], y + dy[i], grid);
            }
        }
    }

    // 计算岛屿数量的主函数
    int numIslands(vector<vector<char>>& grid) {
        // 遍历矩阵的每个元素
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[0].size(); j++) {
                // 如果找到一个未访问的陆地
                if (grid[i][j] == '1') {
                    res++; // 岛屿数量加一
                    dfs(i, j, grid); // 使用DFS标记整个岛屿
                }
            }
        }
        return res; // 返回岛屿数量
    }
};

```

##### [463. 岛屿的周长](https://leetcode.cn/problems/island-perimeter/)

> 思路：寻找一个陆地的周围有多少个水域。

```c++
class Solution {
public:
    int dx[4] = {-1, 1, 0, 0}, dy[4] = {0, 0, -1, 1}; // 上下左右四个方向的向量

    int dfs(int x, int y, vector<vector<int>>& grid) {
        // 如果当前位置在矩阵范围外，或者当前位置为水域（0）
        if (x < 0 || x >= grid.size() || y < 0 || y >= grid[0].size() || grid[x][y] == 0) {
            return 1; // 这条边是岛屿的边缘
        }
        
        // 如果当前格子已经访问过（标记为2），则不计入周长
        if (grid[x][y] == 2) {
            return 0;
        }
        
        // 标记为已访问（使用数字2表示）
        grid[x][y] = 2;
        int perimeter = 0;

        // 遍历四个方向的相邻节点，并累加周长
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            perimeter += dfs(nx, ny, grid);
        }

        return perimeter; // 返回总周长
    }

    int islandPerimeter(vector<vector<int>>& grid) {
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[0].size(); j++) {
                // 找到一个未访问的陆地
                if (grid[i][j] == 1) {
                    return dfs(i, j, grid); // 使用DFS计算并返回岛屿周长
                }
            }
        }

        return 0; // 如果没有找到岛屿，返回0
    }
};

```

##### [695. 岛屿的最大面积](https://leetcode.cn/problems/max-area-of-island/)

> 思路：利用DFS不断的搜索扩大岛屿的面积。每一次入口的DFS能够确认一个岛屿的面积。取最大值即可。如果一个岛屿被访问过，需要将其标记。例如`grid[x][y] = 2; // 标记为已访问，防止重复计算`

```c++
class Solution {
public: 
    int dx[4] = {-1, 1, 0, 0}, dy[4] = {0, 0, -1, 1}; // 上下左右四个方向的向量，用于移动坐标
    int res = 0;   // 记录最大岛屿的面积

    // 深度优先搜索函数，用于计算岛屿面积
    int dfs(int x, int y, vector<vector<int>>& grid) {
        // 检查坐标是否在矩阵范围内，如果超出范围则返回当前面积
        if (x < 0 || x >= grid.size() || y < 0 || y >= grid[0].size())
            return 0;
        
        // 如果当前格子不是陆地（1），也返回当前面积
        if (grid[x][y] != 1)
            return 0;

        int count = 1; // 当前格子为陆地，面积初始化为1
        grid[x][y] = 2; // 标记为已访问，防止重复计算

        // 遍历四个方向的相邻格子，累加面积
        for (int i = 0; i < 4; i++)
            count += dfs(x + dx[i], y + dy[i], grid);

        return count; // 返回累加后的面积
    }

    // 主函数，计算最大的岛屿面积
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        // 遍历矩阵中的每个格子
        for (int i = 0; i < grid.size(); i++)
            for (int j = 0; j < grid[0].size(); j++) {
                // 如果当前格子是未访问的陆地（1）
                if (grid[i][j] == 1) {
                    count = 0; // 重置当前岛屿面积
                    res = max(res, dfs(i, j, grid)); // 计算当前岛屿的面积，并更新最大面积
                }
            }
        return res; // 返回最大岛屿面积
    }
};

```

##### [827. 最大人工岛](https://leetcode.cn/problems/making-a-large-island/)

> 思路：
>
> 1. 利用 DFS 计算出各个岛屿的面积，并标记每个 1（陆地格子）属于哪个岛。
> 2. 遍历每个 0，统计其上下左右四个相邻格子所属岛屿的编号，**去重**后，累加这些岛的面积，更新答案的最大值。
>
> ### 关键点
>
> 1. 如果两个相邻格子属于同一个岛，就会重复计算面积。因此要去重。
> 2. 遍历每个水域，尝试四周是否有相邻的岛屿。
> 3. 利用一个哈希表将岛屿的编号与面积建立映射
> 4. 利用`set`进行去重，相加取最大值。

```c++
class Solution {
public:
    int idx = 2; // 用于给每个岛屿编号，从2开始编号，因为1表示陆地
    int dx[4] = {-1, 1, 0, 0}, dy[4] = {0, 0, -1, 1}; // 四个方向的向量，分别表示上下左右
    unordered_map<int, int> cnt; // 记录岛屿编号与面积的映射

    // 深度优先搜索函数，用于计算岛屿面积，并为岛屿编号
    int dfs(int x, int y, vector<vector<int>>& grid) {
        // 检查坐标是否在矩阵范围内，如果超出范围则返回面积0
        if (x < 0 || x >= grid.size() || y < 0 || y >= grid[0].size())
            return 0;

        // 如果当前格子不是陆地（1），返回面积0
        if (grid[x][y] != 1)
            return 0;

        int area = 1; // 当前格子为陆地，面积初始化为1
        grid[x][y] = idx; // 给当前格子编号，标记为已访问

        // 遍历四个方向的相邻格子，累加面积
        for (int i = 0; i < 4; i++)
            area += dfs(x + dx[i], y + dy[i], grid);

        return area; // 返回累加后的面积
    }

    int largestIsland(vector<vector<int>>& grid) {
        int res = 0; // 初始化最大面积为0

        // 遍历矩阵中的每个格子，计算并记录每个岛屿的面积
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[0].size(); j++) {
                if (grid[i][j] == 1) { // 如果当前格子是陆地
                    idx++; // 给岛屿编号
                    cnt[idx] = dfs(i, j, grid); // 计算岛屿面积，并存储在cnt映射中
                    res = max(cnt[idx], res); // 更新最大岛屿面积
                }
            }
        }

        // 遍历所有的水域，尝试将其变成陆地，并检查能否合并多个岛屿
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[0].size(); j++) {
                if (grid[i][j] == 0) { // 如果当前格子是水域
                    unordered_set<int> neibor; // 用于记录相邻岛屿的编号

                    // 检查四个方向上的相邻格子
                    for (int u = 0; u < 4; u++) {
                        int nx = i + dx[u], ny = j + dy[u];
                        if (nx >= 0 && nx < grid.size() && ny >= 0 && ny < grid[0].size() && grid[nx][ny] >= 2) {
                            neibor.insert(grid[nx][ny]); // 记录相邻岛屿的编号
                        }
                    }

                    int temp_max = 1; // 将当前水域变成陆地后的初始面积为1

                    // 合并相邻的岛屿
                    for (auto num : neibor) {
                        temp_max += cnt[num]; // 累加相邻岛屿的面积
                    }
                    res = max(res, temp_max); // 更新最大岛屿面积
                }
            }
        }
        return res; // 返回最大岛屿面积
    }
};

```

##### [面试题 16.19. 水域大小](https://leetcode.cn/problems/pond-sizes-lcci/)

> 思路：利用`DFS`求连通块大小。方法与求岛屿一致

```c++
class Solution {
public:
    int dx[8]={-1,1,0,0,-1,-1,1,1},dy[8]={0,0,-1,1,1,-1,1,-1};//八个方向
    int dfs(int x,int y,vector<vector<int>> &land){
        if(x<0||x>=land.size()||y<0||y>=land[0].size()||land[x][y]!=0)
            return 0;
        land[x][y]=-1;//标记已被访问
        int area=1;
        for(int i=0;i<8;i++)
            area+=dfs(x+dx[i],y+dy[i],land);
        return area;
    }
    vector<int> pondSizes(vector<vector<int>>& land) {
         vector<int> res;
        for(int i=0;i<land.size();i++)
            for(int j=0;j<land[0].size();j++){
                if(land[i][j]==0){
                    res.emplace_back(dfs(i,j,land));
                }
            }
        sort(res.begin(),res.end());
        return res;
    }
};
```

##### [2658. 网格图中鱼的最大数目](https://leetcode.cn/problems/maximum-number-of-fish-in-a-grid/)

> 思路：DFS 统计每个包含正数的连通块的元素和，最大值即为答案。

```c++
class Solution {
public:
    int dx[8]={-1,1,0,0,-1,-1,1,1},dy[8]={0,0,-1,1,1,-1,1,-1};//八个方向
    int dfs(int x,int y,vector<vector<int>> &grid){
        if(x<0||x>=grid.size()||y<0||y>=grid[0].size()||grid[x][y]<=0)
            return 0;
        int count= grid[x][y];
        grid[x][y]=0;
        for(int i=0;i<4;i++)
            count+=dfs(x+dx[i],y+dy[i],grid);
        return count;
    }
    int findMaxFish(vector<vector<int>>& grid) {
        int res=0;
        for(int i=0;i<grid.size();i++)
            for(int j=0;j<grid[0].size();j++)
                if(grid[i][j]>0)
                    res=max(res,dfs(i,j,grid));
        
        return res;
    }
};
```

##### [1034. 边界着色](https://leetcode.cn/problems/coloring-a-border/)

> 思路：利用DFS遍历判断当前方块是否需要染色，染色条件：1. 位于边界上  2. 与四周的延伸都不相同。
>
> ​		将需要染色的点加入到数组中。

```c++
class Solution {
public:
    static const int N = 60; // 定义最大网格尺寸
    int dx[4] = {-1, 1, 0, 0}; // 上下左右四个方向的 x 坐标变化
    int dy[4] = {0, 0, -1, 1}; // 上下左右四个方向的 y 坐标变化
    vector<pair<int, int>> cnt; // 用于记录需要涂色的边界点
    bool st[N][N] = {false}; // 用于标记已访问的点
    
    // 深度优先搜索函数，标记需要涂色的边界点
    void dfs(int x, int y, vector<vector<int>>& grid, int tar, int color) {
        st[x][y] = true; // 标记该点已访问
        bool isBorder = false; // 标记是否为边界点
        
        for(int i = 0; i < 4; i++) { // 只考虑上下左右四个方向
            int nx = x + dx[i];
            int ny = y + dy[i];
            
            if(nx < 0 || nx >= grid.size() || ny < 0 || ny >= grid[0].size() || grid[nx][ny] != tar) {
                isBorder = true; // 邻居点超出范围或者值不等于目标值，则当前点为边界点
            } else if(!st[nx][ny]) {
                dfs(nx, ny, grid, tar, color); // 继续对邻居点进行DFS
            }
        }
        
        if(isBorder) {
            cnt.push_back({x, y}); // 记录需要涂色的边界点
        }
    }
    
    // 主函数，用于染色指定区域的边界
    vector<vector<int>> colorBorder(vector<vector<int>>& grid, int row, int col, int color) {
        int tar = grid[row][col]; // 获取起点的值
        dfs(row, col, grid, tar, color); // 从起点开始进行DFS
        
        for(auto& point : cnt) { // 遍历所有记录下来的边界点
            grid[point.first][point.second] = color; // 对边界点进行染色
        }
                
        return grid; // 返回染色后的网格
    }
};

```

##### [1020. 飞地的数量](https://leetcode.cn/problems/number-of-enclaves/)

> 思路：根据题目意思，飞地的定义为连通块中没有包含边界的点，因此我们反向思考，我们可以将边界的连通块进行标记，之后进行统计没有标记的连通块即为答案。
>
> ### 关键点
>
> 1. 需要将访问过的点进行标记，防止重复访问。

```c++
class Solution {
public:
    // 定义最大网格大小
    static const int N = 600;
    
    // 标记数组，表示每个格子是否被访问过
    bool st[N][N] = {false};
    
    // 定义四个方向的向量（上下左右）
    int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    int dy[8] = {0, 0, -1, 1, 1, -1, 1, -1};

    // 深度优先搜索函数，用于标记连通的1块
    void dfs(int x, int y, vector<vector<int>> &grid) {
        // 如果超出边界或者遇到0或已标记为-1的格子，则返回
        if (x < 0 || x >= grid.size() || y < 0 || y >= grid[0].size() || grid[x][y] == 0 || grid[x][y] == -1)
            return;

        // 将当前的1标记为-1，表示已经访问过
        st[x][y] = true;
        grid[x][y] = -1;

        // 对四个方向进行DFS搜索
        for (int i = 0; i < 4; i++) {
            dfs(x + dx[i], y + dy[i], grid);
        }
    }

    int numEnclaves(vector<vector<int>>& grid) {
        // 标记边界的连通块编号为-1
        for (int i = 0; i < grid.size(); i++) {
            dfs(i, 0, grid); // 第一列
            dfs(i, grid[0].size() - 1, grid); // 最后一列
        }
                
        // 将第一行与最后一行0连通块标记
        for (int j = 0; j < grid[0].size(); j++) {
            dfs(0, j, grid); // 第一行
            dfs(grid.size() - 1, j, grid); // 最后一行
        }

        int res = 0;
        // 统计内部的未被访问到的1块
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[0].size(); j++) {
                if (grid[i][j] == 1 && !st[i

```



##### [1254. 统计封闭岛屿的数目](https://leetcode.cn/problems/number-of-closed-islands/)

> 思路：被1包围的0连通块，确定一定是在内部，将边界的0连通块标记，之后统计内部的连通块个数，即为被包围的0连通快个数。
>
> ### 具体步骤
>
> 1. **定义方向向量**：首先定义四个方向的向量（上下左右），方便在DFS中进行方向移动。
> 2. **DFS标记函数**：定义一个DFS函数，用于标记与某个0连通的所有0块为2，表示已经访问过。这个函数会递归地访问相邻的0块。
> 3. **标记边界连通块**：遍历第一列和最后一列，以及第一行和最后一行，将与边界连通的所有0块通过DFS标记为2。（边界一定不会是封闭的0连通块）
> 4. **统计封闭岛屿**：遍历整个网格，找到所有没有被标记的0块，计数并通过DFS将其标记，避免重复计数。
> 5. **返回结果**：最终返回计数的封闭岛屿数量。
>
> ### 关键点：
>
> 1. 将访问过的点进行标记，防止重复访问。可以使用bool 数组，或者在原地进行修改

```c++
class Solution {
public:
    // 定义八个方向
    int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    int dy[8] = {0, 0, -1, 1, 1, -1, 1, -1}; 

    static const int N = 150;
    bool st[N][N];

    // 深度优先搜索函数，用于标记连通的0块
    void dfs(int x, int y, vector<vector<int>> &grid) {
        // 如果超出边界或者遇到1或者已经标记过的2，则返回
        if (x < 0 || x >= grid.size() || y < 0 || y >= grid[0].size() || grid[x][y] == 1 || grid[x][y] == 2)
            return;
        
        // 将当前的0标记为2，表示已经访问过
        grid[x][y] = 2;

        // 对四个方向进行DFS搜索
        for (int i = 0; i < 4; i++) {
            dfs(x + dx[i], y + dy[i], grid);
        }
    }

    int closedIsland(vector<vector<int>>& grid) {
        int res = 0;

        // 将第一列与最后一列的0连通块标记
        for (int i = 0; i < grid.size(); i++) {
            dfs(i, 0, grid);
            dfs(i, grid[0].size() - 1, grid);
        }

        // 将第一行与最后一行的0连通块标记
        for (int j = 0; j < grid[0].size(); j++) {
            dfs(0, j, grid);
            dfs(grid.size() - 1, j, grid);
        }

        // 遍历整个网格，找到所有封闭的0块，并进行计数
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[0].size(); j++) {
                if (grid[i][j] == 0) {
                    res++;
                    dfs(i, j, grid); // 将封闭的0块标记
                }
            }
        }

        return res;
    }
};

```

##### [130. 被围绕的区域](https://leetcode.cn/problems/surrounded-regions/)

> 思路：将边界的岛屿进行标记，最后将未标记的进行字符替换，而标记岛屿进行还原。
>
> ### 关键点：
>
> 1. 访问过后的方格记得标记。防止重复访问。

```c++
class Solution {
public:
    //约定位于边界的连通块将其标记为f，
    int dx[8]={-1,1,0,0,-1,-1,1,1},dy[8]={0,0,-1,1,1,-1,1,-1};
    void dfs(int x,int y,vector<vector<char>> &board){
        if(x<0||x>=board.size()||y<0||y>=board[0].size()||board[x][y]!='O')
            return ;
        board[x][y]='F';
        for(int i=0;i<4;i++)
            dfs(x+dx[i],y+dy[i],board);
    }
    void solve(vector<vector<char>>& board) {
        int n=board.size();
        int m=board[0].size();
        //标记四条边界的连通块
        for(int i=0;i<n;i++){
            dfs(i,0,board);
            dfs(i,m-1,board);
        }
        for(int j=0;j<m;j++){
            dfs(0,j,board);
            dfs(n-1,j,board);
        }

        //进行替换将F-》O   O-》X
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++){
                if(board[i][j]=='O')
                    board[i][j]='X';
                else if(board[i][j]=='F')
                    board[i][j]='O';
            }
                
    }
};
```

##### [1905. 统计子岛屿](https://leetcode.cn/problems/count-sub-islands/)

> 思路：采用DFS，如果是g2中的岛屿不包含g1中水域，那么就是一个子岛屿。定义dfs返回当前岛屿是否包含g1中的水域。
>
> ### 关键点
>
> 1. 对于已经访问过的格子，说明此前一定符合条件。直接返回true即可。
> 2. 对于flag变量，统计4个方向的值，不能使用flag=flag&&dfs()。要使用if(!dfs())flag=false;

```c++
class Solution {
public:
    // 上下左右四个方向
    int dx[4] = {-1, 1, 0, 0}, dy[4] = {0, 0, -1, 1}; 
    vector<vector<int>> grid1;
    vector<vector<int>> grid2;
    
    // DFS 搜索函数，检查当前岛屿是否是子岛
    bool dfs(int x, int y, vector<vector<int>>& g) {
        // 边界条件和水域处理
        if (x < 0 || x >= g.size() || y < 0 || y >= g[0].size() || g[x][y] == 0) 
            return true; // 水域或越界都可以直接跳过（视为满足子岛条件）

        // 如果在 grid1 中这个格子不是陆地，那么这个岛屿不是子岛
        if (grid1[x][y] == 0) return false;
        if(g[x][y]==-1)return true;
        // 标记已经访问过这个格子
        g[x][y] = -1;

        bool isSubIsland = true;
        // 遍历四个方向
        for (int i = 0; i < 4; ++i) {
            int nx = x + dx[i], ny = y + dy[i];
            // 递归搜索相邻的格子，且必须保证所有方向都满足条件
            if (!dfs(nx, ny, g)) {
                isSubIsland = false; // 只要有一个方向不满足条件，这个岛屿就不是子岛
            }
        }

        return isSubIsland;
    }

    // 计算子岛的数量
    int countSubIslands(vector<vector<int>>& grid1, vector<vector<int>>& grid2) {
        this->grid1 = grid1;
        this->grid2 = grid2;
        
        int m = grid2.size();
        int n = grid2[0].size();
        int res = 0;

        // 遍历 grid2 中的每个格子，查找岛屿
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                // 如果找到 grid2 中的陆地
                if (grid2[i][j] == 1) {
                    // 如果这个岛屿是子岛，计数加1
                   res+=dfs(i,j,grid2)
                }
            }
        }

        return res;
    }
};

```



#### 网格

##### [1391. 检查网格中是否存在有效路径](https://leetcode.cn/problems/check-if-there-is-a-valid-path-in-a-grid/)

> 思路：由题可知，每个街道总共有两种的探测方向，一共有6种街道，将这6中街道枚举出来，使用`vector<vector<pair<int,int>>> `存储。对于每个街道，尝试向他的两个方向进行延展。什么时候能否延展成功呢？一定是一个相反的方向，例如，左边一个街道有一个向右的方向，右边一个街道有一个向左的方向。即`sum_x`,以及`sum_y`都为`0`。那么说明可以延展成功。访问每个方块后将其标记。最后检查右下角方块是否被访问。
>
> ### 关键点
>
> 1. 6中街道类型要进行枚举
> 2. x分量之和与y分量之和要都为0才能到达。

```c++
class Solution {
public:
    // 每一个街道代表了两个方向，如果来的方向与当前拥有方向相反那么说明可以到达该方格。
    vector<vector<pair<int, int>>> pos{7};
    static const int N = 400;
    bool st[N][N];

    void init_pos() {
        // 初始化每个街道对应的方向
        pos[1] = {{0, -1}, {0, 1}};    // 街道1：左 -> 右
        pos[2] = {{-1, 0}, {1, 0}};    // 街道2：上 -> 下
        pos[3] = {{0, -1}, {1, 0}};    // 街道3：左 -> 下
        pos[4] = {{0, 1}, {1, 0}};     // 街道4：右 -> 下
        pos[5] = {{0, -1}, {-1, 0}};   // 街道5：左 -> 上
        pos[6] = {{0, 1}, {-1, 0}};    // 街道6：右 -> 上
    }

    void dfs(int x, int y, vector<vector<int>>& grid) {
        int n = grid.size();
        int m = grid[0].size();
        if (x < 0 || x >= n || y < 0 || y >= m || st[x][y]) // 超出边界或已经访问
            return;
        st[x][y] = true; // 标记当前已经被访问

        // 探测能够到达的方向
        for (int i = 0; i < 2; i++) {
            // 能够探测到的方格
            int nx = x + pos[grid[x][y]][i].first;
            int ny = y + pos[grid[x][y]][i].second;
            // 监测方格是否能够到达
            if (nx < 0 || nx >= n || ny < 0 || ny >= m || st[nx][ny])
                continue;
            for (int j = 0; j < 2; j++) {
                int sum_x = pos[grid[x][y]][i].first +pos[grid[nx][ny]][j].first;
                int sum_y=  pos[grid[x][y]][i].second  + pos[grid[nx][ny]][j].second;
                if (sum_x == 0&&sum_y==0)
                    dfs(nx, ny, grid);
            }
        }
    }

    bool hasValidPath(vector<vector<int>>& grid) {
        memset(st, false, sizeof(st)); // 初始化访问标记数组
        init_pos(); // 初始化方向数组
        dfs(0, 0, grid); // 从起点 (0, 0) 开始深度优先搜索
        return st[grid.size() - 1][grid[0].size() - 1]; // 返回终点是否被访问
    }
};
```

##### [529. 扫雷游戏](https://leetcode.cn/problems/minesweeper/)

> 思路：利用DFS，不断的更新棋盘，终止条件如下
>
> ### 终止条件
>
> 1. 超出边界，或者以及被访问过（`board[x][y]!=‘E’`）
> 2. 计算当前格子附近的雷数量
>    * 不等于0，停止递归
>    * 等于0，继续递归

```c++
class Solution {
public:
    // 定义八个方向，分别表示上下左右和四个对角线方向
    int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    int dy[8] = {0, 0, -1, 1, 1, -1, 1, -1};

    // 计算当前位置周围有多少雷
    int count_boom(int x, int y, vector<vector<char>>& board) {
        int count = 0;
        // 遍历八个方向，统计周围的雷
        for (int i = 0; i < 8; i++) {
            int nx = x + dx[i], ny = y + dy[i];
            // 判断是否越界以及是否为雷
            if (nx >= 0 && nx < board.size() && ny >= 0 && ny < board[0].size() && board[nx][ny] == 'M')
                count++;
        }
        return count;
    }

    // 深度优先搜索揭示空白区域
    void dfs(int x, int y, vector<vector<char>>& board) {
        // 检查边界和是否已被访问过的非空白格子
        if (x < 0 || x >= board.size() || y < 0 || y >= board[0].size() || board[x][y] != 'E')
            return;
        // 统计该点周围有多少雷
        int count = count_boom(x, y, board);
        if (count != 0) {
            board[x][y] = count + '0'; // 如果周围有雷，更新当前格子为周围雷的数量
        } else {
            board[x][y] = 'B'; // 无雷，更新为 'B'
            // 继续递归揭示周围的格子
            for (int i = 0; i < 8; i++) {
                int nx = x + dx[i], ny = y + dy[i];
                dfs(nx, ny, board);
            }
        }
    }

    // 更新棋盘，根据点击的位置揭示相应区域
    vector<vector<char>> updateBoard(vector<vector<char>>& board, vector<int>& click) {
        int x = click[0], y = click[1];
        // 如果点击到雷，更新为 'X'
        if (board[x][y] == 'M') {
            board[x][y] = 'X';
        } else {
            dfs(x, y, board); // 否则进行深度优先搜索揭示区域
        }
        return board;
    }
};

```



##### [417. 太平洋大西洋水流问题](https://leetcode.cn/problems/pacific-atlantic-water-flow/)

> 题目意思：水往低处流，问那些水能够同时的流进太平洋和大西洋。
>
> 思路：假设太平洋和大西洋的水 从低向高 “攀登”，分别能到达哪些位置，分别用 `p_visited `和 `a_visited `表示。两者的交集就代表能同时流向太平洋和大西洋的位置。从4条边的位置开始往其他地方进行蔓延，条件为当前格子高度`<=`周围格子高度才能进行蔓延。使用两个二维数组 `p_visited` 和 `a_visited`，分别记录太平洋和大西洋的水能从低向高“攀登”到的位置。把能到达的哪些的位置，分别在 `p_visited` 和 `a_visited`标记出来。最后取能够同时到达的点。

```c++
class Solution {
public:
    const int N=300;
    
    // 定义方向数组，表示上下左右四个方向
    int dx[4]={-1, 1, 0, 0};
    int dy[4]={0, 0, -1, 1};
    vector<vector<int>> res; // 存储结果的数组
    vector<int> path{2}; // 临时存储路径的数组，大小为2
    
    // 深度优先搜索函数，用于标记从 (x, y) 出发可以流向的格子
    void dfs(int x, int y, vector<vector<int>> &heights, vector<vector<bool>> &st) {
        st[x][y] = true; // 标记当前格子已访问
        // 检测能否流入周围格子
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i], ny = y + dy[i];
            // 超出边界或已经访问则跳过
            if (nx < 0 || nx >= heights.size() || ny < 0 || ny >= heights[0].size() || st[nx][ny])
                continue;
            // 只有当当前格子高度小于等于邻居格子高度时才能流向邻居格子
            if (heights[x][y] <= heights[nx][ny])
                dfs(nx, ny, heights, st);
        }
    }

    vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
        int n = heights.size(); // 行数
        int m = heights[0].size(); // 列数
        // 初始化两个访问标记数组
        vector<vector<bool>> p_visited(n, vector<bool>(m));
        vector<vector<bool>> a_visited(n, vector<bool>(m));
        path.resize(2, 0); // 初始化路径数组
        
        // 从左边界和右边界出发进行 DFS
        for (int i = 0; i < n; i++) {
            dfs(i, 0, heights, a_visited); // 左边界
            dfs(i, m - 1, heights, p_visited); // 右边界
        }

        // 从上边界和下边界出发进行 DFS
        for (int j = 0; j < m; j++) {
            dfs(0, j, heights, a_visited); // 上边界
            dfs(n - 1, j, heights, p_visited); // 下边界
        }

        // 找出既能流向太平洋又能流向大西洋的格子
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++) {
                if (p_visited[i][j] && a_visited[i][j]) {
                    path[0] = i;
                    path[1] = j;
                    res.emplace_back(path);
                }
            }
        return res;
    }
};

```

> [!NOTE]
>
> 下面为BFS解决问题，多源BFS，由于是在网格中，有点类似于最短路径。



##### [542. 01 矩阵](https://leetcode.cn/problems/01-matrix/)

> 思路：有点类似于最短路径，由于是在图中进行，因此每个节点的边权重为1，有两种思路：1.将所有0入队，往外扩散更新1的距离。2.将所有挨着0的1入队，往外扩散更新1的距离。
>
> 以下为第一种思路
>
> ### 主要思路
>
> - **初始化**：
>   - 初始化距离矩阵 `dist` 为无穷大（用 `INT_MAX` 表示），用以记录每个点到最近 `0` 的最短距离，`0到自己的距离为0`。
>   - 初始化访问标记矩阵 `st` 用于标记已经处理过的点。
>   - 将矩阵中所有 `0` 的位置加入队列 `q`,标记为已经被访问。
> -  **BFS 扩展**：
>   - 从队列中取出一个点，更新其四个相邻点的距离。
>   - 如果他的相邻点为1，并未未被访问过，同时有`dist[nx][ny]>dist[x][y]+1`，表示有更短的路径到0.
> -  **更新距离**：
>   - 对于每个点，使用 BFS 扩展确保所有点的距离都被正确更新。
> -  **返回结果**：
>   - 返回最终计算得到的距离矩阵 `dist`，其中每个点的值表示其到最近 `0` 的最短距离。



```c++
//反向思路：先将挨着0的1距离初始化为1，之后从1开始往其他1进行扩散。
class Solution {
public:
    // dx 和 dy 数组定义了 8 个方向的移动（上下左右及对角线）
    int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1},
        dy[8] = {0, 0, -1, 1, 1, -1, 1, -1};

    vector<vector<int>> updateMatrix(vector<vector<int>>& mat) {
        int n = mat.size(); // 矩阵的行数
        int m = mat[0].size(); // 矩阵的列数
        
        // 初始化距离矩阵，所有位置的初始距离为 0
        vector<vector<int>> dist(n, vector<int>(m, 0));
        
        // 初始化访问标记数组，标记是否已经访问过
        vector<vector<bool>> st(n, vector<bool>(m, false));
        
        queue<PII> q; // 队列用于存储需要处理的点的坐标
        
        // 遍历矩阵，找到所有的 0，并将其邻居的 1 加入队列
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (mat[i][j] == 0) {
                    // 对于每个 0，检查它的四个相邻方向
                    for (int k = 0; k < 4; k++) {
                        int nx = i + dx[k], ny = j + dy[k];
                        // 如果相邻位置在矩阵内且是 1 并且未被访问过
                        if (nx >= 0 && nx < n && ny >= 0 && ny < m && 
                            !st[nx][ny] && mat[nx][ny] == 1) {
                            // 将邻居的 1 的距离初始化为 1
                            dist[nx][ny] = 1;
                            // 将邻居点加入队列，并标记为已访问
                            q.push({nx, ny});
                            st[nx][ny] = true;
                        }
                    }
                }
            }
        }
        
        // 广度优先搜索（BFS）扩展距离
        while (!q.empty()) {
            auto t = q.front();
            q.pop();
            int x = t.first, y = t.second;
            
            // 更新相邻的点的距离
            for (int i = 0; i < 4; i++) {
                int nx = x + dx[i], ny = y + dy[i];
                // 确保相邻点在矩阵内且是 1 并且未被访问过
                if (nx >= 0 && nx < n && ny >= 0 && ny < m &&
                    !st[nx][ny] && mat[nx][ny] == 1) {
                    // 更新距离，并将新点加入队列
                    dist[nx][ny] = dist[x][y] + 1;
                    st[nx][ny] = true;
                    q.push({nx, ny});
                }
            }
        }
        
        return dist; // 返回最终的距离矩阵
    }
};

//正向思路：将所有的0加入队列，然后往1进行扩散，类似最短路径，如果dist[nx][ny]>dist[x][y]+1
class Solution {
public:
    // dx 和 dy 数组定义了 8 个方向的移动（上下左右及对角线），用于探索相邻点
    static const int N = 1e4;
    typedef pair<int, int> PII;
    int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1},
        dy[8] = {0, 0, -1, 1, 1, -1, 1, -1};

    vector<vector<int>> updateMatrix(vector<vector<int>>& mat) {
        int n = mat.size(); // 矩阵的行数
        int m = mat[0].size(); // 矩阵的列数
        
        // 初始化距离矩阵为无穷大（这里用 INT_MAX 表示）
        vector<vector<int>> dist(n, vector<int>(m, INT_MAX));
        
        // 初始化访问标记数组
        vector<vector<bool>> st(n, vector<bool>(m, false));
        
        queue<PII> q; // 队列用于存储需要处理的点的坐标

        // 遍历矩阵，找到所有的 0，并将它们的相邻的 1 点加入队列
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (mat[i][j] == 0) {
                    // 对于每个 0，初始化它的距离为 0，并将它加入队列
                    dist[i][j] = 0;
                    q.push({i, j});
                    st[i][j] = true;
                } else {
                    // 将其他点的初始距离设为无穷大
                    dist[i][j] = 0x3f3f3f3f;
                }
            }
        }
        
        // 广度优先搜索（BFS）扩展距离
        while (!q.empty()) {
            auto t = q.front(); // 取出队列的前端元素
            q.pop();
            int x = t.first, y = t.second;

            // 更新相邻节点的距离
            for (int i = 0; i < 4; i++) {
                int nx = x + dx[i], ny = y + dy[i];
                
                // 确保相邻点在矩阵内且尚未被访问
                if (nx >= 0 && nx < n && ny >= 0 && ny < m && !st[nx][ny]) {
                    // 如果当前点到 0 的距离比以前的距离更短，更新距离
                    if (dist[nx][ny] > dist[x][y] + 1) {
                        dist[nx][ny] = dist[x][y] + 1;
                        st[nx][ny] = true; // 标记为已访问
                        q.push({nx, ny}); // 将更新的点加入队列
                    }
                }
            }
        }

        return dist; // 返回最终计算得到的距离矩阵
    }
};

```

##### [994. 腐烂的橘子](https://leetcode.cn/problems/rotting-oranges/)

> 思路：求最短路径，搜索每个新鲜橘子离腐烂橘子的最短距离。由于是在网格中，因此使用多源BFS较为高效。同时可以省去`dist`数组，以及`st`数组（在原矩阵进行修改）。使用`fresh`记录有多少个新鲜橘子，用`res`记录最少使用了多少分钟。
>
> ### 关键点
>
> 1. 将res初始化为-1，因为，第一层为腐烂橘子，如果都为腐烂橘子，那么需要返回0.
> 2. 最后返回max(res,0)。原因是，在 *grid* 全为 0 的情况下要返回 0，但这种情况下 *ans* 仍为其初始值 −1，所以最后返回的是 max(*ans*,0)。

```c++
class Solution {
public:
    // 将腐烂的橘子全部入队初始化为0，向四周扩散。等价与多源最短路径
    static const int INF=0x3f3f3f3f;
    typedef pair<int,int> PII;
    int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1}; // 八个方向的x轴偏移量
    int dy[8] = {0, 0, -1, 1, 1, -1, 1, -1}; // 八个方向的y轴偏移量
    
    int orangesRotting(vector<vector<int>>& grid) {
        int n = grid.size();   // 获取网格的行数
        int m = grid[0].size(); // 获取网格的列数
        queue<PII> q;          // 用于存储腐烂橘子的队列
        int res = -1;          // 记录腐烂橘子的轮数（时间）
        int fresh = 0;         // 记录新鲜橘子的数量
        
        // 遍历网格，找到所有腐烂的橘子，并统计新鲜橘子的数量
        for(int i = 0; i < n; i++)
            for(int j = 0; j < m; j++)
                if(grid[i][j] == 2) {
                    q.push({i, j}); // 将腐烂的橘子坐标入队
                } else if(grid[i][j] == 1) {
                    fresh++; // 统计新鲜橘子的数量
                }
        
        // 开始BFS
        while(q.size()) {
            int size = q.size();
            res++; // 每一轮增加时间
            while(size--) {
                auto t = q.front(); // 取出队列头部的元素
                q.pop();
                int x = t.first, y = t.second;
                // 遍历四个方向
                for(int i = 0; i < 4; i++) {
                    int nx = x + dx[i], ny = y + dy[i];
                    // 确保新坐标在网格范围内，且该位置为新鲜橘子
                    if(nx < 0 || nx >= n || ny < 0 || ny >= m || grid[nx][ny] != 1)
                        continue;
                    fresh--; // 新鲜橘子数量减少
                    grid[nx][ny] = 2; // 新鲜橘子变成腐烂橘子
                    q.push({nx, ny}); // 将新腐烂的橘子坐标入队
                }
            }
        }
        
        // 如果还有新鲜橘子，返回-1，否则返回腐烂时间
        return fresh != 0 ? -1 : max(res, 0);
    }
};

```

##### [2684. 矩阵中移动的最大次数](https://leetcode.cn/problems/maximum-number-of-moves-in-a-grid/)

> 思路：求第一列元素能够到达的最大列数。利用多源BFS，将第一列元素先入队，进行标记，防止重复访问。利用统计层次的BFS模板即可。
>
> ### 关键点
>
> 1. res初始化为-1，因此第一层为第一列的元素，不能加入计算。
> 2. 将访问过的元素进行标记，防止重复入队

```c++
class Solution {
public:
    typedef pair<int, int> PII;
    int dx[3] = {-1, 0, 1}; // x方向的三个偏移量，分别为上，中，下
    int dy[3] = {1, 1, 1};  // y方向的三个偏移量，均为向右移动一列

    int maxMoves(vector<vector<int>>& grid) {
        int res = -1; // 记录最大步数，初始化为-1
        int n = grid.size(); // 获取网格的行数
        int m = grid[0].size(); // 获取网格的列数
        queue<PII> q; // 用于广度优先搜索的队列
        vector<vector<bool>> st(n, vector<bool>(m, false)); // 记录访问状态的数组
      
        // 将第一列的所有位置入队，并标记为已访问
        for (int i = 0; i < n; i++) {
            q.push({i, 0});
            st[i][0] = true; // 标记为已访问
        }
        
        // 开始广度优先搜索
        while (!q.empty()) {
            res++; // 每完成一轮，步数加一
            int size = q.size();
            while (size--) {
                auto t = q.front();
                q.pop();
                int x = t.first, y = t.second;
                // 遍历三个方向
                for (int i = 0; i < 3; i++) {
                    int nx = x + dx[i], ny = y + dy[i];
                    // 检查新位置是否在边界内，是否已经访问过，是否符合条件
                    if (nx < 0 || nx >= n || ny < 0 || ny >= m || grid[nx][ny] <= grid[x][y] || st[nx][ny])
                        continue;
                    q.push({nx, ny}); // 将新位置入队
                    st[nx][ny] = true; // 标记为已访问
                }
            }
        }
        return res;
    }
};

```

##### [1926. 迷宫中离入口最近的出口](https://leetcode.cn/problems/nearest-exit-from-entrance-in-maze/)

> 思路：寻找走到出口最近需要几步。利用BFS进行探索，本题是单源BFS，因此仅需将起点入队即可，访问一个节点后，将其变成墙。第一次走到的出口，即为最近的出口，同时需要特判，如果起点在边界，要判断不是因为在边界的出口。
>
> ### 关键点
>
> 1. 访问节点需要标记为墙体
> 2. 特判是否是因为起点在边界引起的。

```c++
class Solution {
public:
    typedef pair<int, int> PII;  // 定义坐标对
    int dx[4] = {-1, 1, 0, 0};  // 上下左右四个方向的 x 坐标增量
    int dy[4] = {0, 0, -1, 1};  // 上下左右四个方向的 y 坐标增量

    int nearestExit(vector<vector<char>>& maze, vector<int>& entrance) {
        int n = maze.size();       // 迷宫的行数
        int m = maze[0].size();    // 迷宫的列数
        int res = -1;              // 记录步数，初始为 -1（表示还没有步数）
        
        // 队列用于 BFS 遍历
        queue<PII> q;
        q.push({entrance[0], entrance[1]});  // 将入口位置入队
        maze[entrance[0]][entrance[1]] = '+'; // 将入口位置标记为访问过（墙壁）

        while (!q.empty()) {   // 当队列不为空时继续遍历
            res++;  // 每一层扩展时步数增加
            int size = q.size();  // 当前层的节点数
            while (size--) {      // 遍历当前层的所有节点
                auto t = q.front();  // 取出队首节点
                q.pop();
                int x = t.first, y = t.second;

                for (int i = 0; i < 4; i++) {  // 遍历四个方向
                    int nx = x + dx[i], ny = y + dy[i];  // 计算新位置
                    // 检查是否越界
                    if (nx < 0 || nx >= n || ny < 0 || ny >= m) {
                        // 如果越界且不是入口，返回当前步数（找到出口）
                        if (x != entrance[0] || y != entrance[1])
                            return res;
                        continue;
                    }
                    // 如果是墙壁或已经访问过的节点，跳过
                    if (maze[nx][ny] == '+')
                        continue;
                    // 将新位置加入队列并标记为访问过
                    q.push({nx, ny});
                    maze[nx][ny] = '+';
                }
            }
        }

        return -1;  // 如果遍历结束仍未找到出口，返回 -1
    }
};

```

##### [1162. 地图分析](https://leetcode.cn/problems/as-far-from-land-as-possible/)

> 思路：多源BFS，将所有的陆地加入队列中，标记为已经访问，更新步数即可，res即为最远距离。
>
> ### 关键点
>
> 1. 将访问过的节点标记
> 2. 如果全是海洋或陆地，那么最终返回-1

```c++
class Solution {
public:
    typedef pair<int, int> PII;  // 定义坐标对
    int dx[4] = {-1, 1, 0, 0};  // 上下左右四个方向的 x 坐标增量
    int dy[4] = {0, 0, -1, 1};  // 上下左右四个方向的 y 坐标增量

    int maxDistance(vector<vector<int>>& grid) {
        int res = -1;  // 初始化最远距离为 -1（表示未找到）
        int n = grid.size();  // 网格的行数
        int m = grid[0].size();  // 网格的列数
        queue<PII> q;  // 队列用于广度优先搜索（BFS）

        // 将所有陆地位置入队，并标记为已访问
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                if (grid[i][j] == 1) {
                    q.push({i, j});
                    grid[i][j] = 2;  // 标记为已访问
                }

        // BFS 遍历
        while (!q.empty()) {
            res++;  // 每一层扩展时步数增加
            int size = q.size();  // 当前层的节点数
            while (size--) {  // 遍历当前层的所有节点
                auto t = q.front();
                q.pop();
                int x = t.first, y = t.second;

                for (int i = 0; i < 4; i++) {  // 遍历四个方向
                    int nx = x + dx[i], ny = y + dy[i];  // 计算新位置
                    // 如果新位置越界或不是水域，则跳过
                    if (nx < 0 || nx >= n || ny < 0 || ny >= m || grid[nx][ny] != 0)
                        continue;
                    // 标记新位置为已访问，并加入队列
                    grid[nx][ny] = 2;
                    q.push({nx, ny});
                }
            }
        }

        // 如果 res 为 0（表示没有水域或没有陆地），返回 -1
        return res == 0 ? -1 : res;
    }
};

```

##### [3044. 出现频率最高的质数](https://leetcode.cn/problems/most-frequent-prime/)

> 思路：本题要求乘积为质数，因此我们可以预先处理质数集合，考虑到本题最大的质数不会超过`1e6`。因此我们仅需要筛出范围内的质数即可。同时由于只能沿着一个方向进行，我们需要记录往哪个方向。枚举每个位置的数可以走的路径。如果是质数同时大小超过了10，记录到map中即可。最后遍历map统计处出现次数最大的一个数字

```c++
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <algorithm>
using namespace std;

// 全局常量和变量定义
static const int max_num = 1e6 + 10; // 最大的数范围，用于筛选质数
int primes[max_num+10], cnt; // 质数数组和质数计数器
unordered_set<int> se; // 存储质数的集合，用于快速查找
bool st[max_num+10], is_initial = false; // 标记数组（用于筛选）和初始化标志

// 线性筛法获取所有小于 max_num 的质数
void get_primes() {
    if (!is_initial) { // 确保只初始化一次
        is_initial = true;
        for (int i = 2; i < max_num; i++) {
            if (!st[i]) { // 如果 i 没有被标记为合数，则 i 是质数
                primes[cnt++] = i; // 存储质数
                se.insert(i); // 将质数插入到集合中
            }
            for (int j = 0; primes[j] <= max_num / i; j++) { // 标记质数的倍数为合数
                st[primes[j] * i] = true;
                if (i % primes[j] == 0) break; // 如果 i 能被 primes[j] 整除，则停止，避免重复标记
            }
        }
    }
}

// 解决方案类
class Solution {
public:
    static const int N = 7; // 矩阵的固定大小
    unordered_map<int, int> cnt; // 记录每个质数的出现次数
    int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1}; // 八个可能的移动方向的 x 坐标变化
    int dy[8] = {0, 0, -1, 1, 1, -1, 1, -1}; // 八个可能的移动方向的 y 坐标变化

    // 深度优先搜索（DFS）函数，从 (x, y) 开始沿着方向 d 搜索数字组合
    void dfs(int x, int y, vector<vector<int>> &mat, long long val, int d) {
        // 边界条件检查：如果坐标越界则返回
        if (x < 0 || x >= mat.size() || y < 0 || y >= mat[0].size())
            return;

        // 更新当前数字组合，将当前单元格的数字添加到 val 的末尾
        val = val * 10 + mat[x][y];

        // 如果组合的数字是质数并且大于 10，增加其出现次数
        if (se.contains(val) && val > 10)
            cnt[val]++;

        // 继续沿着相同的方向 d 递归搜索
        int nx = x + dx[d], ny = y + dy[d];
        dfs(nx, ny, mat, val, d);
    }

    // 主函数：查找矩阵中出现频率最高的质数
    int mostFrequentPrime(vector<vector<int>>& mat) {
        get_primes(); // 初始化质数集合
        int max_count = 0, most_frequent_prime = -1; // 用于存储最大出现次数和对应的质数

        // 从每个矩阵点 (i, j) 开始，在所有 8 个方向上执行 DFS 搜索
        for (int i = 0; i < mat.size(); i++) {
            for (int j = 0; j < mat[0].size(); j++) {
                for (int k = 0; k < 8; k++) {
                    dfs(i, j, mat, 0, k); // 初始值 `val` 为 `0`，方向为 `k`
                }
            }
        }

        // 遍历质数的计数表，找到出现次数最多的质数
        for (auto t : cnt) {
            int frequency = t.second; // 质数出现的频率
            int prime_number = t.first; // 当前质数
            if (frequency > max_count) {
                max_count = frequency;
                most_frequent_prime = prime_number;
            } else if (frequency == max_count) {
                most_frequent_prime = max(most_frequent_prime, prime_number); // 如果出现次数相同，选择较大的质数
            }
        }

        return most_frequent_prime; // 返回出现频率最高的质数
    }
};

```

##### [1765. 地图中的最高点](https://leetcode.cn/problems/map-of-highest-peak/)

> 思路：多源BFS，题目要求相邻高度差至多为1，又要构成最大的高度值，那么利用BFS即可，每次往外扩散1层。能够得到最大的高度

```c++
class Solution {
public:
    int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1}; // 八个方向的x轴偏移量
    int dy[8] = {0, 0, -1, 1, 1, -1, 1, -1}; // 八个方向的y轴偏移量
    typedef pair<int,int> PII;
    vector<vector<int>> highestPeak(vector<vector<int>>& isWater) {
        int m=isWater.size();
        int n=isWater[0].size();

        vector<vector<bool>> st(m+1,vector<bool>(n+1,false));
        queue<PII> q;
        int res=0;
        //将所有的水域入队
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(isWater[i][j]==1){
                    q.push({i,j});
                    st[i][j]=true;
                    isWater[i][j]=0;
                }
            }
        }
        while(q.size()){
            res++;
            int size=q.size();
            while(size--){
                auto t=q.front();
                q.pop();
                int x=t.first,y=t.second;
                for(int i=0;i<4;i++){
                    int nx=x+dx[i],ny=y+dy[i];
                    if(nx<0||nx>=m||ny<0||ny>=n||st[nx][ny])continue;
                    q.push({nx,ny});
                    isWater[nx][ny]=res;
                    st[nx][ny]=true;
                }
            }
        }
        return isWater;
    }
};
```

##### [934. 最短的桥](https://leetcode.cn/problems/shortest-bridge/)

> 思路：DFS编号+多源BFS，题目意思为，寻找两个岛屿之间最短的路径
>
> 做法：利用DFS给两个岛屿都编上号，利用BFS，将其中一个岛屿入队，寻找第二个岛屿的最短路径。
>
> ### 关键点
>
> 1. 利用DFS编号
> 2. 利用多源BFS搜索另外一个岛屿

```c++
class Solution {
public:
    int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1}; // 八个方向的x轴偏移量
    int dy[8] = {0, 0, -1, 1, 1, -1, 1, -1}; // 八个方向的y轴偏移量
    typedef pair<int,int>PII ;
    int cnt=2;
    void dfs(int x,int y,vector<vector<int>> &g){
        if(x<0||x>=g.size()||y<0||y>=g[0].size()||g[x][y]!=1)
            return ;
        //标记陆地
        g[x][y]=cnt;
        for(int i=0;i<4;i++){
            int nx=x+dx[i],ny=y+dy[i];
            dfs(nx,ny,g);
        }
    }
    int shortestBridge(vector<vector<int>>& grid) {
        int m=grid.size();
        int n=grid[0].size();
        //vector<vector<bool>> st(m+1,vector<bool>(n+1,false));

        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(grid[i][j]==1){
                    dfs(i,j,grid);
                    cnt++;
                }
            }
        }
        queue< PII > q;
         int res=0;
        //将所有的陆地2入队
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(grid[i][j]==2){
                    q.push({i,j});
                }
            }
        }

        while(q.size()){
            res++;
            int size=q.size();
            while(size--){
                auto t=q.front();
                q.pop();
                int x=t.first,y=t.second;
                for(int i=0;i<4;i++){
                    int nx=x+dx[i],ny=y+dy[i];
                    if(nx<0||nx>=m||ny<0||ny>=n||grid[nx][ny]==-1)continue;
                    if(grid[nx][ny]==3)return res-1;
                    grid[nx][ny]=-1;
                    q.push({nx,ny});
                }
            }
        }

        return 0;
    }
};
```

##### [2146. 价格范围内最高排名的 K 样物品](https://leetcode.cn/problems/k-highest-ranked-items-within-a-price-range/)

> 思路：BFS+排序  题目要求价值最高的排序序列，距离越近，优先级越高，因此，采用BFS，向外不断的扩散，扩散到的一层，那么这一层的距离都是相同的，只需要比较，价格，行，列，来进行排序即可。因此采用一个`vector<tuple<int,PII>>` 存储这一层所有物品的价值和坐标，一层结束之后，按照要求进行排序，将排序后的结果挑选加入到res中即可

```c++
class Solution {
public:
    typedef pair<int,int> PII;
    int dx[4] = {0, 0,-1, 1};  // 四个方向的x轴偏移量
    int dy[4] = {-1, 1, 0, 0}; // 四个方向的y轴偏移量

    static bool cmp(const tuple<int, PII>& a, const tuple<int, PII>& b) {
        if (get<0>(a) != get<0>(b)) return get<0>(a) < get<0>(b);  // 价格升序
        if (get<1>(a).first != get<1>(b).first) return get<1>(a).first < get<1>(b).first; // 行号升序
        return get<1>(a).second < get<1>(b).second;  // 列号升序
    }

    vector<vector<int>> highestRankedKItems(vector<vector<int>>& grid, vector<int>& pricing, vector<int>& start, int k) {
        int m = grid.size(), n = grid[0].size();
        vector<vector<bool>> visited(m, vector<bool>(n, false));

        queue<PII> q;
        q.push({start[0], start[1]});
        visited[start[0]][start[1]] = true;

        vector<vector<int>> res;
        vector<tuple<int, PII>> candidates;

        while (!q.empty()) {
            int size = q.size();
            vector<tuple<int, PII>> temp;

            while (size--) {
                auto [x, y] = q.front();
                q.pop();

                // 判断是否符合价格区间
                if (grid[x][y] >= pricing[0] && grid[x][y] <= pricing[1]) {
                    temp.push_back({grid[x][y], {x, y}});
                }

                // 四个方向的搜索
                for (int i = 0; i < 4; ++i) {
                    int nx = x + dx[i], ny = y + dy[i];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && !visited[nx][ny] && grid[nx][ny] != 0) {
                        q.push({nx, ny});
                        visited[nx][ny] = true;
                    }
                }
            }

            // 按照价格、坐标排序
            sort(temp.begin(), temp.end(), cmp);

            for (auto& [price, coord] : temp) {
                res.push_back({coord.first, coord.second});
                if (res.size() == k) return res;
            }
        }

        return res;
    }
};

```

##### [1293. 网格中的最短路径](https://leetcode.cn/problems/shortest-path-in-a-grid-with-obstacles-elimination/)

> 思路：`BFS`，题目要求最多消除K个障碍物的情况下，从左上角走到右下角的最短路径，利用`BFS`，维护状态为 `{x,y,r}`表示当前到`x,y`这个点，所剩余的可以移除障碍物的次数为r，利用一个三维的数组`vector<vector<vector<bool>>`  记录达到一个点 x,y所剩余不同的次数r的情况。
>
> ### 关键点
>
> 1. 为什么不能在原有的矩阵上进行标记访问？ 原因是，到达某一个点，所剩余的移除障碍物的次数可能有所不同，如果在原数组上进行标记，那么可能会丢失最优的路径。因此要使用一个三维的`bool`类型数组来标记当前状态`{x,y,r}`是否已被访问

```c++
class Solution {
public:
    typedef tuple<int, int, int> PIII; // {x, y, r} r表示剩余可消耗障碍物数
    int dx[4] = {0, 0, -1, 1};  // 四个方向的x轴偏移量
    int dy[4] = {-1, 1, 0, 0};  // 四个方向的y轴偏移量
    
    int shortestPath(vector<vector<int>>& grid, int k) {
        int m = grid.size();
        int n = grid[0].size();
        if (m == 1 && n == 1) return 0;  // 起点就是终点的特殊情况
        
        // 访问标记数组，三维 visited[x][y][r] 表示在 (x, y) 剩余 r 次消除时是否访问过
        vector<vector<vector<bool>>> visited(m, vector<vector<bool>>(n, vector<bool>(k + 1, false)));
        
        queue<PIII> q;
        q.push({0, 0, k});
        visited[0][0][k] = true;
        
        int res = 0; // 用于跟踪步数
        
        while (!q.empty()) {
            res++;
            int size = q.size();
            while (size--) {
                auto [x, y, r] = q.front();
                q.pop();
                
                // 尝试四个方向移动
                for (int i = 0; i < 4; ++i) {
                    int nx = x + dx[i], ny = y + dy[i];
                    
                    // 越界检查
                    if (nx < 0 || nx >= m || ny < 0 || ny >= n) continue;
                    
                    // 如果到达终点
                    if (nx == m - 1 && ny == n - 1) return res;
                    
                    int nr = r - grid[nx][ny]; // 如果是障碍物，消耗一次
                    if (nr < 0) continue; // 没有足够的消除次数，无法继续
                    
                    // 如果在 (nx, ny) 剩余 nr 次消除时未访问过，则继续 BFS
                    if (!visited[nx][ny][nr]) {
                        visited[nx][ny][nr] = true;
                        q.push({nx, ny, nr});
                    }
                }
            }
        }
        
        // 如果无法到达终点，返回 -1
        return -1;
    }
};

```

##### [1036. 逃离大迷宫](https://leetcode.cn/problems/escape-a-large-maze/)

> 思路：由于此题的网格盘是一个`1e6*1e6`的数据量，巨大，因此，我们使用常规的bfs到达终点会超时，思考与blocked块有关系。可以推出，给定障碍物能够围成的最大面积为`limit=n*(n-1)/2`。一共有三种情况：1.从起点出发能够到达终点。2.从终点出发能够到达起点。3.二者而能够到达的方块个数都超过了limit
>
> 思路2：用离散化建图

```c++
class Solution {
public:
    int dx[4] = {0, 0, 1, -1}, dy[4] = {1, -1, 0, 0};
    typedef pair<int, int> PII;
    int limit;
    set<PII> blocked;
    int tx, ty;
    set<PII> st;
    
    bool bfs(int a, int b) {
        queue<PII> q;
        q.push({a, b});
        st.insert({a, b});
        
        while (!q.empty()) {
            auto [x, y] = q.front();
            q.pop();
            
            if (x == tx && y == ty && a != tx && b != ty) return true;
            
            for (int i = 0; i < 4; i++) {
                int nx = x + dx[i], ny = y + dy[i];
                PII next(nx, ny);
                
                if (nx < 0 || nx >= 1e6 || ny < 0 || ny >= 1e6 || 
                    st.count(next) || blocked.count(next)) continue;
                
                if (st.size() > limit) return true;
                
                q.push(next);
                st.insert(next);
            }
        }    
        return false;
    }
    
    bool isEscapePossible(vector<vector<int>>& blocked_points, vector<int>& source, vector<int>& target) {
        if (blocked_points.empty()) return true;
        
        for (const auto& t : blocked_points) {
            this->blocked.emplace(t[0], t[1]);
        }

        limit = blocked_points.size() * (blocked_points.size() - 1) / 2;
        tx = target[0], ty = target[1];
        
        // 先尝试从起点到终点
        bfs(source[0], source[1]);
        if (st.count({tx, ty})) return true;
        
        st.clear();
        return bfs(source[0], source[1]) && bfs(target[0], target[1]);
    }
};


class Solution {
public:
    static const int N=1e6;
    int dx[4]={1,-1,0,0},dy[4]={0,0,1,-1};
    typedef pair<int,int> PII;
    bool isEscapePossible(vector<vector<int>>& blocked, vector<int>& source, vector<int>& target) {
        //离散化
        vector<int> rows;
        vector<int> cols;
        unordered_map<int,int> row_map;
        unordered_map<int,int> col_map;
        for(auto &t:blocked){
            rows.push_back(t[0]);
            cols.push_back(t[1]);
        }
        rows.push_back(source[0]);
        rows.push_back(target[0]);
        cols.push_back(source[1]);
        cols.push_back(target[1]);

        sort(rows.begin(),rows.end());
        sort(cols.begin(),cols.end());
        //去重
        rows.erase(unique(rows.begin(),rows.end()),rows.end());
        cols.erase(unique(cols.begin(),cols.end()),cols.end());
        
        int row_id=(rows[0]==0?0:1);
        row_map[rows[0]]=row_id;
        for(int i=1;i<rows.size();i++){
            row_id+=(rows[i]==rows[i-1]+1)?1:2;//确定相邻点的距离，如果距离超过1，那么要设置距离为2表示二者之间有距离
            row_map[rows[i]]=row_id;
        }

        //判断最后一个点是否是边界
        if(rows[rows.size()-1]!=N-1)row_id++;
        int col_id=(cols[0]==0?0:1);
        col_map[cols[0]]=col_id;
        for(int i=1;i<cols.size();i++){
            col_id+=(cols[i]==cols[i-1]+1)?1:2;
            col_map[cols[i]]=col_id;
        }
        if(cols[cols.size()-1]!=N-1)col_id++;
        vector<vector<int>> grid(row_id+1,vector<int> (col_id+1,0));
        //建图
        for(auto &t:blocked){
            int x=row_map[t[0]],y=col_map[t[1]];
            grid[x][y]=1;
        }
        grid[row_map[source[0]]][col_map[source[1]]]=2;
        grid[row_map[target[0]]][col_map[target[1]]]=2;


        int bx=row_map[source[0]],by=col_map[source[1]];
        int ex=row_map[target[0]],ey=col_map[target[1]];
        vector<vector<bool>> st(row_id+1,vector<bool> (col_id+1,false));
        queue<PII> q;
        q.push({bx,by});
        st[bx][by]=true;

        while(q.size()){
            int size=q.size();
            while(size--){
                auto [x,y]=q.front();
                q.pop();

                if(x==ex&&y==ey)return true;
                for(int i=0;i<4;i++){
                    int nx=x+dx[i],ny=y+dy[i];
                    if(nx<0||nx>row_id||ny<0||ny>col_id||st[nx][ny]||grid[nx][ny]==1)continue;
                    st[nx][ny]=true;
                    q.push({nx,ny});
                }
            }
        }
        return false;
        

    }
};
```



#### 最小生成树

##### [1584. 连接所有点的最小费用](https://leetcode.cn/problems/min-cost-to-connect-all-points/)

> 两种思路：`prim`算法或者`kruskal`算法（并查集）
>
> ### 关键点
>
> 1. 查看下标从`1`还是`0`开始
> 2. 将点映射为一维的点，`points`数组中下标即为他们各自的点编号。

```c++
class Solution {
public:
    static const int N = 1e3 + 10, INF = 0x3f3f3f3f;
    int g[N][N]; // 邻接矩阵，存储点之间的距离
    int dist[N]; // 到最小生成树的最短距离
    bool st[N];  // 标记是否已经加入最小生成树

    // Prim算法计算最小生成树的总权重
    int prim(int n) {
        int res = 0;
        memset(dist, 0x3f, sizeof dist); // 初始化dist数组为无穷大
        memset(st, 0, sizeof st); // 初始化st数组为false
        dist[0] = 0; // 从节点0开始构建最小生成树

        for (int i = 0; i < n; i++) {
            int t = -1;
            // 找到当前未加入最小生成树的点中距离最小的点
            for (int j = 0; j < n; j++)
                if (!st[j] && (t == -1 || dist[j] < dist[t]))
                    t = j;
            // 如果i不是0且dist[t]为无穷大，说明图不连通，返回无穷大
            if (i && dist[t] == INF) return INF;
            if (i) res += dist[t]; // 将该点加入最小生成树，累加其权重

            // 更新未加入最小生成树的点到已加入点的最小距离
            for (int j = 0; j < n; j++)
                dist[j] = std::min(dist[j], g[t][j]);
            st[t] = true; // 标记该点已加入最小生成树
        }
        return res; // 返回最小生成树的总权重
    }

    // 计算连接所有点的最小成本
    int minCostConnectPoints(vector<vector<int>>& points) {
        memset(g, 0x3f, sizeof g); // 初始化邻接矩阵为无穷大
        int n = points.size();
        
        // 构建邻接矩阵，存储点之间的曼哈顿距离
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int w = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1]);
                g[i][j] = g[j][i] = w;
            }
        }
        
        // 调用Prim算法计算最小生成树的总权重
        int s = prim(n);
        return s;
    }
};


//kruskal算法
class Solution {
public:
    static const int N=1e3+10,INF=0x3f3f3f3f;
   // int dist[N];//到最小生成树的最小距离
   // bool st[N];//标记是否已经加入最小生成树
    int p[N];//祖宗节点
    struct Edge{
        int a, b, w;
        //升序排列
        bool operator <(const Edge &W)const{
            return w<W.w;
        }
    };
    int find(int x){
        if(p[x]!=x)p[x]=find(p[x]);
        return p[x];
    }
    int kruskal(int m,int n,vector<Edge> &edges){
        sort(edges.begin(),edges.end());
        int res=0,cnt=0;
        for(int i=0;i<n;i++)p[i]=i;
        //添加n-1条边进集合
        for(int i=0;i<m;i++){
            int a=edges[i].a,b=edges[i].b,w=edges[i].w;
            a=find(a),b=find(b);
            if(a!=b){
                p[a]=b;
                res+=w;
                cnt++;
            }
    }
        if(cnt<n-1)return INF;
        return res;

    }
    int minCostConnectPoints(vector<vector<int>>& points) {
       
        vector<Edge> edges;
        for(int i=0;i<points.size();i++){
            for(int j=i+1;j<points.size();j++){
                int w=abs(points[i][0]-points[j][0])+abs(points[i][1]-points[j][1]);
                edges.emplace_back(Edge{i,j,w});
           
            }
        }
        int n=points.size();
        int m=edges.size();
        int s=kruskal(m,n,edges);
        return s;
    }
};
```

### 二分图

#### 二分图判断

##### [785. 判断二分图](https://leetcode.cn/problems/is-graph-bipartite/)

> 思路：本题是检测所给无向图是否是一个二分图，共有两种解决办法，1. DFS染色法。2.并查集
>
> 染色法思路：判断当前点是否已被染色，若没有进行染色，则进行染色，同时遍历他的邻点，如果邻点为染色，则染成跟当前点相反颜色，继续往下递归，如果dfs返回false代表图中有两个相邻的点颜色相同。如果已经染色，判断是否跟当前点颜色一致，一致返回false。最后如果检测合格返回true
>
> 并查集思路：将一个点的邻点进行合并，同时要判断邻点跟当前点是否处于一个集合中，如果处于一个集合中返回false，最后如果检查成功返回true

```c++
class Solution {
public:
    static const int N=200;
    int p[N];//祖宗节点
    int find(int x){
        if(p[x]!=x)p[x]=find(p[x]);
        return p[x];
    }
    bool isBipartite(vector<vector<int>>& graph) {
        //初始化并查集
        int n=graph.size();
        for(int i=0;i<n;i++)p[i]=i;
        //将当前点的所有相邻的点放到同一个集合中。
        for(int i=0;i<n;i++){
            for(auto j:graph[i]){
                //要先判断当前点没有跟邻点处于同一个集合中。
                if(find(i)==find(j))return false;
                else
                    p[find(graph[i][0])]=find(j);
            }
        }
        return true;
    }
};



class Solution {
public:
    static const int N=200,M=1e5;
    int h[N],e[M],ne[M],idx;
    int color[N];//-1代表为染色，0白色，1黑色

    bool dfs(int u,int c){
        color[u]=c;
        for(int i=h[u];i!=-1;i=ne[i]){
            int j=e[i];
            if(color[j]==-1){
                //当前节点未被染色
                if(dfs(j,!c)==false)return false;
            }else{
                if(color[j]==c)
                    return false;
            }

        }
        return true;
    }
    void add(int a,int b){
        e[idx]=b;
        ne[idx]=h[a];
        h[a]=idx++;
    }
    bool isBipartite(vector<vector<int>>& graph) {
        memset(color,-1,sizeof color);
        memset(h,-1,sizeof h);
        //初始化邻接表
        for(int i=0;i<graph.size();i++){
            for(int j=0;j<graph[i].size();j++){
                add(i,graph[i][j]);
                add(graph[i][j],i);
            }
        }
        int n=graph.size();
        for(int i=0;i<n;i++){
            if(color[i]==-1){
                //未染色
                if(dfs(i,0)==false)return false;
            }
        }
        return true;
    }
};
```

#####  [886. 可能的二分法](https://leetcode.cn/problems/possible-bipartition/)

> 思路：本题是二分图加了一层伪装，实质上还是判断是否为一个二分图。1.染色法，2.并查集

```c++
class Solution {
public:
    // 定义常量 N 表示最大节点数，M 表示最大边数
    static const int N = 2500, M = 1e5 + 100;
    int h[N], e[M], ne[M], idx; // 邻接表：h 存储每个节点的邻接表头节点索引，e 存储边的终点，ne 存储下一条边的索引
    int p[N]; // 并查集数组 p

    // 并查集的查找函数，带路径压缩
    int find(int x) {
        if (p[x] != x) p[x] = find(p[x]);
        return p[x];
    }

    // 向邻接表中添加一条边 a -> b
    void add(int a, int b) {
        e[idx] = b;
        ne[idx] = h[a];
        h[a] = idx++;
    }

    // 判断是否可以将图分成两组
    bool possibleBipartition(int n, vector<vector<int>>& dislikes) {
        // 初始化邻接表
        memset(h, -1, sizeof h);
        for (int i = 0; i < dislikes.size(); i++) {
            int a = dislikes[i][0], b = dislikes[i][1];
            add(a, b); // 添加 a -> b
            add(b, a); // 添加 b -> a (无向图)
        }

        // 初始化并查集
        for (int i = 1; i <= n; i++) p[i] = i;

        // 遍历每个节点，检查是否可以二分
        for (int i = 1; i <= n; i++) {
            if (h[i] == -1) continue; // 如果没有邻接点，跳过该节点
            int first = e[h[i]]; // 获取第一个邻接点
            for (int j = h[i]; j != -1; j = ne[j]) {
                int k = e[j]; // 当前边的终点
                if (find(i) == find(k)) return false; // 如果当前节点和其邻接点在同一个集合，说明有奇环，返回 false
                else p[find(first)] = find(k); // 将所有邻接点合并到同一个集合
            }
        }
        return true; // 如果没有冲突，返回 true
    }
};


class Solution {
public:
    static const int N = 2500, M = 1e5 + 100; // 定义常量 N 表示最大节点数，M 表示最大边数
    int h[N], e[M], ne[M], idx; // 邻接表相关数组：h 存储每个节点的邻接表头节点索引，e 存储边的终点，ne 存储下一条边的索引
    int color[N]; // 颜色数组，用于染色法判断二分图

    // 向邻接表中添加一条边 a -> b
    void add(int a, int b) {
        e[idx] = b;
        ne[idx] = h[a];
        h[a] = idx++;
    }

    // 深度优先搜索，u 为当前节点，c 为当前节点的颜色
    bool dfs(int u, int c) {
        color[u] = c; // 将当前节点染色
        for (int i = h[u]; i != -1; i = ne[i]) { // 遍历所有邻接点
            int j = e[i]; // 取出邻接点
            if (color[j] == -1) { // 如果邻接点未被染色
                if (!dfs(j, !c)) return false; // 递归染色，如果出现冲突返回 false
            } else {
                if (color[j] == c) // 如果邻接点已被染色且颜色相同
                    return false; // 返回 false
            }
        }
        return true; // 如果没有冲突，返回 true
    }

    // 判断是否可以将图分成两组
    bool possibleBipartition(int n, vector<vector<int>>& dislikes) {
        memset(h, -1, sizeof h); // 初始化邻接表
        memset(color, -1, sizeof color); // 初始化颜色数组

        // 根据 dislikes 数组构建邻接表
        for (int i = 0; i < dislikes.size(); i++) {
            int a = dislikes[i][0], b = dislikes[i][1];
            add(a, b);
            add(b, a);
        }

        // 遍历每个节点
        for (int i = 1; i <= n; i++) {
            if (color[i] == -1) // 如果当前节点未被染色
                if (!dfs(i, 0)) return false; // 进行深度优先搜索染色，如果出现冲突返回 false
        }
        return true; // 如果没有冲突，返回 true
    }
};
```

## 12.并查集

### [547. 省份数量](https://leetcode.cn/problems/number-of-provinces/)

> 思路：本题计算的是，不同集合的数量，因此采用朴素并查集即可具体过程如下，先初始化并查集，同时将连通的两个城市加入到一个集合中，最后统计不同集合的个数。
>
> ### 具体过程
>
> * **初始化**：创建并查集的 `p` 和 `size` 数组，每个城市初始时自成一集。
> *  **合并**：遍历 `isConnected` 矩阵，对于每个相连的城市对 `(i, j)`，调用 `merge` 函数进行合并。
> *  **统计**：遍历所有城市，统计根节点等于自身的城市数量，即为连通分量的数量。

```c++
class Solution {
public:
    static const int N = 310;
    int p[N], size[N];
   int find(int x) {
        if (p[x] != x)
            p[x] = find(p[x]);
        return p[x];
    }

    void merge(int a, int b) {
        a = find(a), b = find(b);
        if (a != b) {
            size[b] += size[a];
            p[a] = b;
        }
    }
    int findCircleNum(vector<vector<int>>& isConnected) {
        int n=isConnected.size();
        // 初始化
        for (int i = 0; i <n; i++) {
            p[i] = i;
            size[i] = 1;
        }

        for (int i = 0; i < n; i++) 
            for(int j=0;j<n;j++){
               if(isConnected[i][j]==1){
                //说明i与j是同一个集合
                merge(i,j);
               }
            }
        //记录城市数量
        int cnt=0;
        for(int i=0;i<n;i++){
            if(find(i)==i)
                cnt++;
        }    
        return cnt;
    }
};
```

### [684. 冗余连接](https://leetcode.cn/problems/redundant-connection/)

> 思路：题目翻译过来就是求，如果两个点在同一个集合中了，又出现一条边让两个点再一个集合中。
>
> ### 具体过程
>
> - **初始化并查集**：使用 `p` 数组存储每个节点的祖宗节点，初始时每个节点的祖宗节点为其自身。
> - **查找操作**：`find` 函数用于查找节点的祖宗节点，并进行路径压缩，以加速后续的查找操作。
> - **合并操作**：`merge` 函数用于合并两个节点所在的集合。如果两个节点的祖宗节点相同，则说明这条边是多余的，否则合并这两个集合。
> - **处理所有边**：遍历所有边，调用 `merge` 函数进行处理，如果发现多余的边，则将其存储在 `res` 中。
> - **返回结果**：返回存储多余边的 `res` 数组。

```c++
class Solution {
public:
    static const int N = 1100;
    int p[N]; // 祖宗数组，用于存储每个节点的祖宗节点

    // 查找节点 x 的祖宗节点，并进行路径压缩
    int find(int x) {
        if (p[x] != x)
            p[x] = find(p[x]);
        return p[x];
    }

    vector<int> res; // 存储多余的边

    // 合并两个节点 a 和 b 所在的集合
    void merge(int a, int b) {
        int tempa = a;
        int tempb = b;
        a = find(a); // 找到 a 的祖宗节点
        b = find(b); // 找到 b 的祖宗节点
        if (a != b) {
            p[a] = b; // 如果祖宗节点不同，则合并两个集合
        } else {
            res = {tempa, tempb}; // 如果祖宗节点相同，则表示这条边是多余的
        }
    }

    // 找到图中多余的边
    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        int n = edges.size();
        // 初始化祖宗数组
        for (int i = 1; i <= n; i++) 
            p[i] = i;
        // 遍历所有边，进行合并操作
        for (int i = 0; i < n; i++) {
            int a = edges[i][0], b = edges[i][1];
            merge(a, b);
        }
        return res; // 返回多余的边
    }
};

```

### [990. 等式方程的可满足性](https://leetcode.cn/problems/satisfiability-of-equality-equations/)

> 思路：本题依旧是并查集的应用，首先将所有的字母看做单独节点，将相等的合并，最后遍历不等的，如果在一个集合，代表不满足要求
>
> ### 具体过程
>
> - **初始化并查集**：使用 `p` 数组存储每个节点的祖宗节点，初始时每个节点的祖宗节点为其自身。
> - **查找操作**：`find` 函数用于查找节点的祖宗节点，并进行路径压缩，以加速后续的查找操作。
> - **合并操作**：`merge` 函数用于合并两个节点所在的集合。
> - **处理相等方程**：遍历所有 "==" 方程，调用 `merge` 函数合并相等的节点。
> - **处理不等方程**：遍历所有 "!=" 方程，检查是否有矛盾（即不等的节点在同一集合中）。如果发现矛盾，则返回 false。
> - **返回结果**：如果没有发现矛盾，返回 true，表示所有方程可以同时成立。

```c++
class Solution {
public:
    static const int N = 500;
    int p[N]; // 并查集数组，用于存储每个节点的祖宗节点

    // 查找节点 x 的祖宗节点，并进行路径压缩
    int find(int x) {
        if (p[x] != x) p[x] = find(p[x]);
        return p[x];
    }

    // 合并两个节点 a 和 b 所在的集合
    void merge(int a, int b) {
        a = find(a);
        b = find(b);
        if (a != b)
            p[a] = b;
    }

    // 判断方程是否可能同时成立
    bool equationsPossible(vector<string>& equations) {
        // 初始化并查集数组
        for (int i = 0; i < N; i++)
            p[i] = i;

        // 处理所有 "==" 方程
        for (auto const& s : equations) {
            int a = s[0] - 'a'; // 将字符转换为对应的索引值
            int b = s[3] - 'a';
            string op = s.substr(1, 2); // 获取操作符 "==" 或 "!="
            if (op == "==")
                merge(a, b); // 合并相等的节点
        }

        // 处理所有 "!=" 方程
        for (auto const& s : equations) {
            int a = s[0] - 'a'; // 将字符转换为对应的索引值
            int b = s[3] - 'a';
            string op = s.substr(1, 2); // 获取操作符 "==" 或 "!="
            if (op == "!=" && find(a) == find(b))
                return false; // 如果发现相等的节点在同一集合中，则矛盾
        }
        
        return true; // 如果没有发现矛盾，返回 true
    }
};

```

### [1202. 交换字符串中的元素](https://leetcode.cn/problems/smallest-string-with-swaps/)

> 思路：由于处于同一个联通块内的字符可以任意的交换，因此采用并查集来进行处理。除了单独使用`P[N]`记录每个元素的祖先节点外，还需要使用一个数组来记录同一个联通块内的字符，以及各个联通分量块内的索引（每个字符仅能使用一次）。
>
> 大致的思路为：1.初始化 2.合并字符  3.联通分量排序。 4.组合成字符串
>
> ### 关键点：
>
> 1. 由于每个字符，仅能使用一次，因此，我们要记录一下，每个联通块内的字符使用到哪个位置了

```c++
class Solution {
public:
    vector<int> p;  // 父节点数组

    int find(int x) {
        return p[x] == x ? x : p[x] = find(p[x]);
    }

    void merge(int x, int y) {
        p[find(y)] = find(x);
    }

    string smallestStringWithSwaps(string s, vector<vector<int>>& pairs) {
        int n = s.length();
        // 初始化并查集
        p.resize(n);
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }

        // 合并所有可交换的位置
        for (const auto& pair : pairs) {
            merge(pair[0], pair[1]);
        }

        // 收集每个连通分量
        vector<string> groups(n);  // 存储每个连通分量的字符
        vector<int> indices(n);    // 存储每个连通分量的索引
        
        // 将字符按连通分量分组
        for (int i = 0; i < n; i++) {
            int root = find(i);
            groups[root] += s[i];
        }

        // 对每个连通分量的字符进行排序
        for (int i = 0; i < n; i++) {
            if (!groups[i].empty()) {
                sort(groups[i].begin(), groups[i].end());
            }
        }

        // 构造结果
        string result = s;
        for (int i = 0; i < n; i++) {
            int root = find(i);
            result[i] = groups[root][indices[root]++];
        }

        return result;
    }
};
```

### [1061. 按字典序排列最小的等效字符串](https://leetcode.cn/problems/lexicographically-smallest-equivalent-string/)

> 思路：由于题目要找出`baseStr`的最小等价字符串，同时在联通块内的字符都是等价的，因此我们采用并查集来进行处理。由于仅需要找到对应位置等价字符的最小一个即可，因此我们在进行合并的时候可以选择最小的字符作为根。
>
> 大致思路：
>
> 1. 初始化
> 2. 合并字符（按照较小的字符作为根）
> 3. 构造结果字符串（以baseStr中的字符来进行找联通块，如果没有，只能以自己作为结果字符）

```c++
class Solution {
public:
    static const int N = 150;  // 支持ASCII码范围0-199
    int p[N];
    
    // 查找根节点（带路径压缩）
    int find(int x) {
        if (x != p[x]) {
            p[x] = find(p[x]);  // 路径压缩，将当前节点直接连接到根节点
        }
        return p[x];
    }
    
    // 合并两个集合
    void merge(int a, int b) {
        a = find(a);  // 找到a的根节点
        b = find(b);  // 找到b的根节点
        if (a < b) {  // 优化：让较小的字符作为根节点
            p[b] = a;
        } else {
            p[a] = b;
        }
    }
    
    string smallestEquivalentString(string s1, string s2, string baseStr) {
        // 初始化并查集，每个字符初始时指向自己
        for (int i = 0; i < N; i++) {
            p[i] = i;
        }
        
        // 合并等价字符
        for (int i = 0; i < s1.size(); i++) {
            merge(s1[i], s2[i]);
        }
        
        // 构建结果字符串
        string res;
        for (char ch : baseStr) {
            // 找到当前字符所在集合的根节点（即最小等价字符）
            res += (char)find(ch);
        }
        
        return res;
    }
};
```



## 13.动态规划

### 背包问题



#### 0-1背包问题

##### [494. 目标和](https://leetcode.cn/problems/target-sum/)

> 思路：题目要求表达式结果等于`target`的数量。也即求方案数。总共有两种思考思路
>
> 1. 回溯：对于每个数`nums[i]`。总共有两种选择方案，选择正数，选择负数。那么最终就会有2^n种选择方案。使用一个`count`.统计总共有多少种方案。在进行回溯的过程中，记录一下已经选择的个数，如果发现有`sum==target`那么将`count`加1.否则的话继续选择当前位置数字的正数和负数。
>
> 2. 动态规划：
>
>    **问题转化**：
>
>    - 原问题是找到一种方法，将数组 `nums` 中的正数和负数分成两部分，使得它们的和的差等于 `target`。
>    - 假设正数的和为 `p`，负数绝对值的和为 `sum - p`（其中 `sum` 是 `nums` 的总和）。
>    - 根据题意，有 `p - (sum - p) = target`。化简得 `2p = sum + target`，即 `p = (sum + target) / 2`。
>
>    **条件判断**：
>
>    - 为了保证` p`是一个合法的和，需要满足以下两个条件：
>      1. `p` 必须是非负数，即 `p >= 0`。
>      2. `sum + target` 必须是偶数，即 `(sum + target) % 2 == 0`。
>    - 如果这两个条件有任意一个不满足，则没有符合条件的方案，返回 0。
>
>    **0-1 背包问题**：
>
>    - 转化后的问题实际上是在数组 `nums` 中找到一个子集，使得该子集的和为 `p`。
>    - 这可以通过0-1背包问题来解决：每个元素只能选一次，求和为 `p` 的子集个数。
>
>    

```c++
//回溯方法
class Solution {
public:
    int count=0;
    void dfs(int i,int sum,int target,vector<int> &nums){
        if(i==nums.size()){
            if(sum==target)
                count++;
            return ;
        }else{
            //选择正的该元素和负的该元素
            dfs(i+1,sum+nums[i],target,nums);
            dfs(i+1,sum-nums[i],target,nums);
        }
    }
    int findTargetSumWays(vector<int>& nums, int target) {
        dfs(0,0,target,nums);
        return count;
    }
};

//动态规划方法
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int target) {
        int n = nums.size(); // 数组长度
        int sum = accumulate(nums.begin(), nums.end(), 0) + target; // 计算 (sum + target)
        int m = sum / 2; // 背包容量
        
        // 如果 sum 为负数或 sum 不是偶数，则没有合法方案
        if (sum < 0 || sum % 2 != 0) {
            return 0;
        }
        
        // 动态规划数组 f，表示从前 i 个数中选择，且总和不超过 j 的方案数
        vector<vector<int>> f(n + 1, vector<int>(m + 1, 0));
        
        // 初始化：从前 0 个数中选择，总和不超过 0 的方案数为 1
        f[0][0] = 1;
        
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= m; j++) {
                // 不选第 i 个数
                f[i][j] = f[i - 1][j];
                // 选第 i 个数，前提是 j >= nums[i - 1]
                if (j >= nums[i - 1]) {
                    f[i][j] += f[i - 1][j - nums[i - 1]];
                }
            }
        }
        
        // 返回最终答案，从前 n 个数中选择，总和为 m 的方案数
        return f[n][m];
    }
};

```

##### [2915. 和为目标值的最长子序列的长度](https://leetcode.cn/problems/length-of-the-longest-subsequence-that-sums-to-target/)

> 思路：本题是从`n`个元素中选择元素，每个元素仅能选择一次。因此看作是0-1背包问题。由题意，定义`f[i][j]`为从前i个物品中选择，体积恰好等于`j`的最大长度。属性为`max`。`f[i][j]`集合总共可以分为两种情况，选择`nums[i-1]`（第i个物品）。不选择`nums[i-1]`。不选择的情况等价于从`1~i-1`中选择元素体积不超过j因此表示为`f[i-1][j]`。选择的情况表示为，如果当前背包容量能够装下则装，装不下只能选择不选。表示为`if(j>=nums[i-1]) f[i-1][j-nums[i-1]]+1`。最终的表达式为`f[i][j]=max(f[i-1][j],f[i-1][j-nums[i-1]]+1)`
>
> ### 关键点
>
> 1. 数组初始化为`INT_MIN`

```c++
class Solution {
public:
    int lengthOfLongestSubsequence(vector<int>& nums, int target) {
        int n = nums.size(); // 数组的长度
        int m = target; // 目标和
        // 创建一个二维数组 f，用于存储动态规划的状态。f[i][j] 表示从前 i 个数字中选择，和恰好等于 j 的最长子序列长度。
        vector<vector<int>> f(n + 1, vector<int>(m + 1, INT_MIN));

        // 初始条件：从前 0 个数字中选择，和为 0 的子序列长度为 0
        f[0][0] = 0;

        // 遍历所有的数字
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= m; j++) {
                // 不选择当前数字 nums[i-1]
                f[i][j] = f[i-1][j];
                // 如果可以选择当前数字 nums[i-1]
                if (j >= nums[i-1]) {
                    // 更新选择当前数字后的子序列长度
                    f[i][j] = max(f[i][j], f[i-1][j-nums[i-1]] + 1);
                }
            }
        }

        // 返回结果，如果没有找到和为 target 的子序列，返回 -1
        return f[n][m] <= INT_MIN / 2 ? -1 : f[n][m];
    }
};

```

##### [416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/)

> 思路：要进行分割两个和相等的子集，那么必须确保数组sum为偶数，则问题转化为在数组中挑选元素和恰好为`sum/2`。采用0-1背包解决问题。属性为`sum`。分为当前元素选择和不选择。最终的状态转移方程为`f[i][j]=f[i-1][j]+f[i-1][j-nums[i-1]]`。

```c++
class Solution {
public:
    static const int MOD=1e9+7;
    bool canPartition(vector<int>& nums) {
        int n = nums.size();
        int sum = accumulate(nums.begin(), nums.end(), 0);
        
        // 如果总和是奇数，不可能分成两个和相等的子集
        if (sum % 2 != 0) return false;

        int target = sum / 2;

        // 创建一个二维数组 f，f[i][j] 表示从前 i 个数字中选择，总和恰好等于 j 的方案数
        // 使用 long long 类型来避免溢出问题
        vector<vector<int>> f(n + 1, vector<int>(target + 1, 0));

        // 初始条件：从前 0 个数字中选择，和为 0 的方案数为 1（不选任何数字）
        f[0][0] = 1;

        // 动态规划填表
        for (int i = 1; i <= n; ++i) {
            for (int j = 0; j <= target; ++j) {
                f[i][j] = f[i-1][j]; // 不选择当前数字
                if (j >= nums[i-1]) {
                    f[i][j] += f[i-1][j-nums[i-1]]; // 选择当前数字
                }
                f[i][j]%=MOD;
            }
        }

        // 打印 f 数组的内容（调试用）
        // for (int i = 0; i <= n; ++i) {
        //     for (int j = 0; j <= target; ++j)
        //         cout << f[i][j] << " ";
        //     cout << endl;
        // }

        // 如果有至少一种方案可以分割成两个和为 target 的子集，则返回 true
        return f[n][target] > 0;
    }
};

```

##### [2787. 将一个数字表示成幂的和的方案数](https://leetcode.cn/problems/ways-to-express-an-integer-as-sum-of-powers/)

> 思路：将数字看做是一个一个的物品，将物品的幂看做是体积。采用0-1背包即可。`f[i][j]`表示从`1-i`中选择，体积恰好为`j`的方案数。属性为`sum`。分为选择和不选择。状态转移方程为 ` f[i][j]=f[i-1][j]+f[i-1][j-nums[i-1]]`

```c++
class Solution {
public:
    static const int MOD = 1e9 + 7;
    int numberOfWays(int n, int x) {
        // 创建二维数组 f
        vector<vector<int>> f(n + 1, vector<int>(n + 1, 0));
        
        // 初始条件
        f[0][0] = 1;

        // 计算幂次并更新 f 数组
        for (int i = 1; i <= n; ++i) {
            long long power = pow(i, x); // 预计算当前 i 的幂次值
            for (int j = 0; j <= n; ++j) {
                f[i][j] = f[i-1][j]; // 不选择当前元素 i
                if (j >= power) {
                    f[i][j] = (f[i][j] + f[i-1][j-power]) % MOD; // 选择当前元素 i
                }
            }
        }
        return f[n][n];
    }
};

```

#### 完全背包问题

##### [322. 零钱兑换](https://leetcode.cn/problems/coin-change/)

> 思路：题目要求数字的个数。因此初始化`f[0][0]`为`0`表示0组成0不需要使用数字。定义`f[i][j]`为从前`i`个物品中选择，体积恰好为`j`的**最小**物品个数。本题可以重复进行选择，因此，使用完全背包模型。总共有k种方案，分别是，不选，选一个，选两个，选k个。最终的转移方程为`f[i][j]=min(f[i-1][j],f[i][j-v]+1)`
>
> ### 关键点
>
> 1. 如果是最大值最小值问题，最后的返回值需要和特殊边界比较一下，如果不符合返回`-1`

```c++
class Solution {
public:
    const int INF=0x3f3f3f3f;  // 定义一个较大的数表示无穷大

    int coinChange(vector<int>& coins, int amount) {
        int n = coins.size();  // 硬币的数量
        int m = amount;  // 目标金额
        
        // 初始化二维数组f，大小为(n+1) x (m+1)，所有值初始化为INF
        vector<vector<int>> f(n+1, vector<int>(m+1, INF));
        
        // 初始化边界条件：不使用任何硬币时，凑成0金额所需的硬币数为0
        f[0][0] = 0;
        
        // 遍历所有硬币
        for(int i = 1; i <= n; i++) {
            // 遍历所有可能的金额
            for(int j = 0; j <= m; j++) {
                // 不选择当前硬币的情况
                f[i][j] = f[i-1][j];
                
                // 选择当前硬币的情况
                if(j >= coins[i-1]) {
                    f[i][j] = min(f[i][j], f[i][j - coins[i-1]] + 1);
                }
            }
        }
        
        // 判断最终结果，如果无法凑成amount，返回-1
        return f[n][m] >= INF/2 ? -1 : f[n][m];
    }
};

```

##### [518. 零钱兑换 II](https://leetcode.cn/problems/coin-change-ii/)

> 思路：本题求的是组合数，也就是方案数，定义`f[0][0]=1`，表示从1-0中选择数字组成0的方案数仅有一个，也就是空集。
>
> 定义`f[i][j]`为从1-i中选择物品，体积恰好等于j的方法个数。属性为`sum`。状态转移方程为` f[i][j]=f[i-1][j]+f[i][j-v];`。

```c++
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        int n = coins.size(); // 硬币的种类数
        int m = amount; // 目标金额
        // 定义一个二维数组 f，f[i][j] 表示用前 i 种硬币构成金额 j 的方案数
        vector<vector<long long>> f(n + 1, vector<long long>(m + 1, 0));
        f[0][0] = 1; // 用 0 种硬币构成金额 0 的方案数为 1，即空集

        // 遍历每种硬币
        for (int i = 1; i <= n; i++) {
            // 遍历每个可能的金额
            for (int j = 0; j <= m; j++) {
                f[i][j] = f[i - 1][j]; // 不选择当前硬币
                if (j >= coins[i - 1]) {
                    // 选择当前硬币，增加方案数
                    f[i][j] = f[i][j] + f[i][j - coins[i - 1]];
                }
            }
        }
        // 返回用前 n 种硬币构成金额 m 的方案数
        return f[n][m];
    }
};

```

##### [279. 完全平方数](https://leetcode.cn/problems/perfect-squares/)

> 思路：本题求的是数字的个数，定义`f[0][0]=0`，表示0组成0不需要数字。定义`f[i][j]`为从1-i中选择物品，物品的体积恰好等于j的最小物品个数。总共有k种方案，不选，选一个，选两个，……选k个。状态转移方程为`f[i][j]=min(f[i-1][j],f[i][j-v]+1)`。
>
> ### 关键点
>
> 1. 对于求最大最小，返回时需要与极端值作对比，不符合返回-1；

```c++
class Solution {
public:
    const int INF = 0x3f3f3f3f; // 定义一个较大的常数表示无穷大
    int numSquares(int n) {
        int max_n = sqrt(n); // 计算 n 的平方根，表示最大的平方数
        // 定义一个二维数组 f，f[i][j] 表示用前 i 个平方数构成 j 所需的最少数量的平方数
        vector<vector<int>> f(max_n + 1, vector<int>(n + 1, INF));

        f[0][0] = 0; // 用 0 个平方数构成 0 需要 0 个数

        // 遍历每个平方数
        for (int i = 1; i <= max_n; i++) {
            int power = pow(i, 2); // 计算当前数 i 的平方
            // 遍历每个可能的和
            for (int j = 0; j <= n; j++) {
                f[i][j] = f[i - 1][j]; // 不选择当前平方数
                if (j >= power) {
                    // 选择当前平方数，更新最小数量
                    f[i][j] = min(f[i][j], f[i][j - power] + 1);
                }
            }
        }
        return f[max_n][n]; // 返回用前 max_n 个平方数构成 n 所需的最少数量的平方数
    }
};

```

### 线性DP问题

#### 最长递增子序列问题（LIS）

##### [300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/)

> 思路：题目要求最长递增的子序列，子序列可以不连续，定义f[i]表示为以nums[i]结尾的集合。属性为`max`.初始化所有的f为1，因为子序列长度为1。可以划分为i个集合，分别为以`0`结尾，以`1`结尾，以`2`结尾，….以`i-1`结尾。如果当前的`nums[i]>nums[j]`代表可以以当前元素结尾。即`f[i]=f[j]+1`

```c++
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n=nums.size();
        vector<int> f(n+1,1);//每个位置递增子序列最少为1

        for(int i=1;i<n;i++)
            for(int j=0;j<i;j++)
            if(nums[i]>nums[j])
                f[i]=max(f[i],f[j]+1);
        return *max_element(f.begin(),f.end());
    }
};
```

##### [673. 最长递增子序列的个数](https://leetcode.cn/problems/number-of-longest-increasing-subsequence/)

> 思路：此题为朴素LIS的升级版本，计算最长递增子序列的和朴素一致，需要定义一个`f[i]`数组代表以`nums[i]`结尾的最长递增子序列长度。同时定义一个`g[i]`定义为以`nums[i]`结尾的最长递增子序列长度的个数。都划分为i个集合。以`0`结尾，以`1`结尾，以`2`结尾，….以`i-1`结尾。如果满足` nums[j]<nums[i]`，说明 `nums[i]` 可以接在 `nums[j] `后面形成上升子序列，此时使用` f[j] `更新 `f[i]`，即有` f[i]=f[j]+1`。在转移`f[i]`的过程中还需要考虑`g[i]`的转移。如果满足 `nums[j]<nums[i]`，说明 `nums[i]` 可以接在 `nums[j]` 后面形成上升子序列，这时候对 `f[i]` 和 `f[j]+1` 的大小关系进行分情况讨论：
>
> 1. 如果有`f[i]<f[j]+1`：说明`f[i]`会被`f[j]+1`进行覆盖。最长子序列长度也会发生改变。因此LIS的个数`g[i]`也就等于`g[j]`
> 2. 如果有`f[i]==f[j]+1`：说明还有不同的以`nums[i]`结尾长度为`f[i]`的子序列.此时LIS的个数为`g[i]+=g[j]`（方案数相加）
>
> 最后使用`res`进行统计`f[i]==max+len`的个数`res+=g[i];`

```c++
class Solution {
public:
    int findNumberOfLIS(vector<int>& nums) {
        int n = nums.size();
        if (n == 0) return 0;

        vector<int> f(n, 1); // f[i] 表示以 nums[i] 结尾的最长递增子序列的长度
        vector<int> g(n, 1); // g[i] 表示以 nums[i] 结尾的最长递增子序列的数量

        int max_len = 1; // 最长递增子序列的长度

        // 动态规划计算
        for (int i = 1; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                if (nums[i] > nums[j]) {
                    if (f[i] < f[j] + 1) {
                        f[i] = f[j] + 1;
                        g[i] = g[j]; // 重置 g[i] 为 g[j] 的数量
                    } else if (f[i] == f[j] + 1) {
                        g[i] += g[j]; // 增加 g[i] 的数量
                    }
                }
            }
            max_len = max(max_len, f[i]);
        }

        // 统计所有长度为 max_len 的递增子序列的数量
        int count = 0;
        for (int i = 0; i < n; ++i) {
            if (f[i] == max_len) {
                count += g[i];
            }
        }

        return count;
    }
};

```

##### [2826. 将三个组排序](https://leetcode.cn/problems/sorting-three-groups/)

> 思路：此题为`LIS`问题的换皮，原题要找删除最小个数，使得剩余数组元素为非递减的有序序列。翻译过来即找到最长的非递减的子序列长度。用数组长度减去最长飞递减子序列长度即为答案

```c++
class Solution {
public:
    int minimumOperations(vector<int>& nums) {
        int n = nums.size();
        vector<int> f(n, 1); // f[i] 表示以 nums[i] 结尾的最长非递减子序列的长度
        int max_len = 1; // 最长非递减子序列的长度

        // 动态规划计算最长非递减子序列的长度
        for (int i = 1; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                if (nums[i] >= nums[j]) {
                    f[i] = max(f[i], f[j] + 1);
                }
            }
            max_len = max(max_len, f[i]);
        }

        // 需要的最少操作次数是数组长度减去最长非递减子序列的长度
        return n - max_len;
    }
};

```

##### [368. 最大整除子集](https://leetcode.cn/problems/largest-divisible-subset/)

> 思路：此题同样是`LIS`的一个变种问题。只不过由最长递增子序列变成了最长的互余子序列。最后题目要求返回最长的互余子序列。因此要额外定义一个`g[i]`表示当前以`nums[i]`结尾的互余子序列是从下标`g[i]`转移过来的。
>
> ### 关键点：
>
> 1. 由于要求序列中所有元素互余，如果序列不是有序的，那么我们需要从当前子序列开头进行判断，非常耗费时间。所以选择进行排序。这样只需要跟子序列的最大值进行比较即可。

```c++
class Solution {
public:
    vector<int> largestDivisibleSubset(vector<int>& nums) {
        int n=nums.size();
        vector<int> f(n,1);
        vector<int> g(n,-1);//记录最大的子序列下标
        int max_len=1;
        sort(nums.begin(),nums.end());
        for(int i =1;i<n;i++){
            for(int j=0;j<i;j++){
                if((nums[i]%nums[j]==0)&&f[i]<f[j]+1){
                    f[i]=max(f[i],f[j]+1);
                    g[i]=j;//记录从哪转移过来的
                    max_len=max(max_len,f[i]);
                }
            }
        }

        for(int x:f)
        cout<<x<<" ";
        vector<int> res;
        //取出最后一个下标往前遍历。
        int idx=-1;
        for(int i=0;i<n;i++)
            if(f[i]==max_len)
                idx=i;
        res.emplace_back(nums[idx]);

        for(int i=g[idx];i!=-1;i=g[i]){
            res.emplace_back(nums[i]);
        }
        return res; 
    }
};
```

#### 最长公共子序列问题（LCS）

##### [583. 两个字符串的删除操作](https://leetcode.cn/problems/delete-operation-for-two-strings/)

> 思路：本题为LCS的换皮，相当于，求出LCS的长度后，用`m+n-2*LCS`。
>
> 思路2：`LCS`做一点细微的改变定义，定义为f[i][j]表示为所有由A前i个字符以及B前j个字符构成相同串需要删除个数。属性为`min`。分为两种情况，1.当前字符相同即`A[i]==B[j]`。就不用删除字符。那么删除最小的个数等于`f[i-1][j-1]`。2. 当前字符不同，那么有可能删除`A[i]`字符或者删除`B[j]`字符。因此`f[i][j]=min(f[i-1][j],f[i][j-1])+1`。含义为，删除字符需要的最小个数。状态转移方程为 `f[i][j]=min(f[i-1][j-1],min(f[i-1][j],f[i][j-1])+1)`需要初始化第一行第一列

```c++
//思路1
class Solution {
public:
    static const int N=510;
    int f[N][N];//定义为所有由A前i个字符，B前j个字符构成的子序列
    int minDistance(string word1, string word2) {
        int m=word1.size();
        int n=word2.size();
        int res=0;
        for(int i=0;i<m;i++)
            for(int j=0;j<n;j++){
                f[i+1][j+1]=max(f[i][j+1],f[i+1][j]);
                if(word1[i]==word2[j])
                f[i+1][j+1]=max(f[i+1][j+1],f[i][j]+1);
                res=max(res,f[i+1][j+1]);
            }
            return m+n-2*res;
    }
};

//思路2
class Solution {
public:
    static const int N=510;
    int f[N][N];//定义为所有由A前i个字符，B前j个字符子串相等需要的操作次数  属性为min
    int minDistance(string word1, string word2) {
        int m=word1.size();
        int n=word2.size();
        int res=0;
        for(int i=0;i<=m;i++)
            f[i][0]=i;
        for(int j=0;j<=n;j++)
            f[0][j]=j;
        for(int i=1;i<=m;i++)
            for(int j=1;j<=n;j++){
               f[i][j]=min(f[i-1][j],f[i][j-1])+1;
               if(word1[i-1]==word2[j-1])
               f[i][j]=min(f[i-1][j-1],f[i][j]);
            }
        return f[m][n];
    }
};
```

##### [712. 两个字符串的最小ASCII删除和](https://leetcode.cn/problems/minimum-ascii-delete-sum-for-two-strings/)

> 思路1：考虑LCS，先将两个字符串的和求出，最后求出最长的LCS，通过之后用m+n-s*LCS即可
>
> 思路2：LCS定义微调，定义`f[i][j]`表示全部由s1的前i个字符，s2的前j个字符构成相同字串删除的ASCCI吗和，属性为`MIN`
>
> 与上题一致，LCS分为四种情况，其中两个全部删除的包含在删除i或者删除j内，因此仅需要考虑三种情况即可。状态转移方程为
>
> `f[i][j]=min(f[i-1][j-1],min(f[i][j-1]+s2[j],f[i-1][j]+s1[i])`
>
> ### 关键点
>
> 1. 初始化，需要对`f[i][0]`,`f[0][j]`进行初始化，表示，当B没有字符，让二者相等需要删除A的ASCII码总和，以及A没有字符，删除B的ASCII码总和。
> 2. 小技巧：出现了i-1,j-1，集体像右边挪一位。变成`f[i+1][j+1]=min(f[i][j],min(f[i+1][j]+s2[j],f[i][j+1]+s1[i])`

```c++
class Solution {
public:
    int minimumDeleteSum(string s1, string s2) {
        int m=s1.size();
        int n=s2.size();
        vector<vector<int>> f(m+1,vector<int>(n+1,0));
        for(int i=0;i<m;i++){
             f[i+1][0]=f[i][0]+s1[i];
        }
        for(int j=0;j<n;j++)
            f[0][j+1]=f[0][j]+s2[j];
        
        for(int i=0;i<m;i++)
            for(int j=0;j<n;j++){
                //删除二者其中一个
                f[i+1][j+1]=min(f[i][j+1]+s1[i],f[i+1][j]+s2[j]);
                if(s1[i]==s2[j])
                f[i+1][j+1]=min(f[i+1][j+1],f[i][j]);
            }

        return f[m][n];
    }
};
```



##### [72. 编辑距离](https://leetcode.cn/problems/edit-distance/)

> 思路：定义`f[i][j]`表示为前A的前i个字符，变成B的前j个字符需要的操作次数，属性为`MIN`。集合可以划分为4个子集，分别是，删，改，增，不动。删表示为A的前i-1个字符和B的前j个字符已经匹配了，即表示为`f[i-1][j]+1`。改表示为A的前i-1个字符已经和B的前j-1个字符匹配完成，表示为`f[i-1][j-1]+1`。增表示为A的前i个字符已经和B的前j-1个字符匹配完成，即表示为`f[i][j-1]+1`。不动表示为A的前i个字符和B的前j个字符匹配完成，等价与`f[i-1][j-1]`。最后的状态转移方程为
>
> `f[i][j]=min(min(f[i][j-1]+1,f[i-1][j]+1),f[i-1][j-1]+1,f[i-1][j-1])`。注：不动这个操作不一定存在，只有当`A[i]==B[j]`时才存在。
>
> ### 关键点
>
> 1. 初始化：需要将`f[i][0]`以及`f[0][j]`进行初始化，表示当B没有字符时应该删除A的字符操作数，以及A没有字符，应该增加的字符操作数。
> 2. 小技巧：由于出现了`i-1,j-1`因此我们可以整体往右挪一位。即变成`f[i+1][j+1]=min(min(f[i+1][j]+1,f[i][j+1]+1),f[i][j]+1,f[i][j])`

```c++
class Solution {
public:
    int minDistance(string word1, string word2) {
        int m = word2.size();
        int n = word1.size();

        // 定义二维动态规划表格，大小为 (n+1) x (m+1)
        vector<vector<int>> f(n + 1, vector<int>(m + 1, 0));
        f[0][0] = 0; // 初始条件：两个空字符串的编辑距离为0

        // 填充第一列：只能通过删除操作将word1转换为空字符串
        for (int i = 1; i <= n; i++)
            f[i][0] = i;

        // 填充第一行：只能通过插入操作将空字符串转换为word2
        for (int j = 1; j <= m; j++)
            f[0][j] = j;

        // 动态规划填表
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                // 计算增加和删除操作的最小值
                f[i][j] = min(f[i][j - 1], f[i - 1][j]) + 1;
                // 计算替换操作的最小值
                f[i][j] = min(f[i][j], f[i - 1][j - 1] + 1);
                // 如果当前字符相同，则可以不进行替换
                if (word1[i - 1] == word2[j - 1])
                    f[i][j] = min(f[i][j], f[i - 1][j - 1]);
            }
        }

        return f[n][m]; // 返回将word1转换为word2的最小编辑距离
    }
};

```

##### [115. 不同的子序列](https://leetcode.cn/problems/distinct-subsequences/)

> 思路：定义一个f[i][j]表示全部由A的前i个字符，构成B子串1-j的方案数，属性为`sum`。集合可以分为两个子集，分别是，选当前`A[i]`字符，和不选当前`A[i]`字符。不选意味着A的前1~i-1个字符跟B的1-j个字符匹配。那么表示为`f[i-1][j]`选择的话表示为A的1-i个字符跟B的1-j个字符都匹配，方案数表示为`f[i-1][j-1]`状态转移方程为`f[i][j]=f[i-1][j]+f[i-1][j-1]`注意：选择的情况不一定存在。
>
> ### 关键点
>
> 1. 初始化，第一行第一列`f[i][0]`都初始化为1，表示，由前i个字符构成空串的方案数只有1个。`f[0][j]`初始化为0，表示，空串无法构成字符

```c++
class Solution {
public:
    static const int MOD=1e9+7;
    int numDistinct(string s, string t) {
        int m=s.size();
        int n=t.size();
        vector<vector<int>> f(m+10,vector<int>(n+10,0));

        //初始化第一行第一列
        f[0][0]=1;//空串构成空串有一种方案
        for(int i=1;i<=m;i++)//不是空串构成空串的方案数，仅能选择空串
        f[i][0]=1;
        for(int j=1;j<=n;j++)//空串构成非空，方案数为0；
        f[0][j]=0;

        for(int i=1;i<=m;i++)
            for(int j=1;j<=n;j++){
                f[i][j]=f[i-1][j];
                if(s[i-1]==t[j-1])
                f[i][j]=f[i][j]+f[i-1][j-1];
                f[i][j]%=MOD;
            }
        return f[m][n];
    }
};
```

##### [1035. 不相交的线](https://leetcode.cn/problems/uncrossed-lines/)

> 思路：此题为LCS的换皮问题，定义f[i][j]为所有由A的前i个数字以及B的前j个数字构成的匹配数，属性为`MAX`。与LCS类似，分为四种情况考虑，00,01,10,11，其中00情况包含在01和10中，因此我们仅考虑三种情况即可。状态转移方程为`f[i][j]=max(f[i-1][j],f[i][j-1],f[i-1][j-1]+1)`
>
> ### 关键点
>
> 1. 重复计算：对于求最大值最小值问题，我们定义状态，及时重复也可以的

```c++
class Solution {
public:
    int maxUncrossedLines(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size();
        int n = nums2.size();
        vector<vector<int>> f(m + 1, vector<int>(n + 1, 0));

        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                f[i][j] = max(f[i-1][j], f[i][j-1]);
                if (nums1[i-1] == nums2[j-1]) {
                    f[i][j] = max(f[i][j], f[i-1][j-1] + 1);
                }
            }
        }
        return f[m][n];
    }
};
```

##### [1458. 两个子序列的最大点积](https://leetcode.cn/problems/max-dot-product-of-two-subsequences/)

> 思路：与LCS类似，定义f[i][j]为全部由A的前i个数字以及B的前j个数字构成的点积，属性为`MAX`，如果没有负数点积的话，跟LCS一样分为4中情况即可，即00,01,10,11，其中00情况包含在01和10中，考虑三种情况即可，状态转移方程为`f[i][j]=max(f[i][j-1],f[i-1][j],f[i-1][j-1]+x*y)`。但是由于有负数的情况，前面的可能点积可能小于只选择当前的点积，因此需要额外的考虑一种情况，即单独的`x*y`。状态转移方程为`f[i][j]=max(f[i-1][j],f[i][j-1],f[i-1][j-1]+x*y,x*y);`
>
> ### 关键点
>
> 1. 此题由于有负数，同时求最大值，因此状态数组`f`需要初始化为`-INF`

```c++
class Solution {
public:
    static const int INF=0x3f3f3f3f;
    int maxDotProduct(vector<int>& nums1, vector<int>& nums2) {

        int n=nums1.size();
        int m=nums2.size();

        vector<vector<int>> f(n+2,vector<int>(m+2,-INF));


        for(int i=1;i<=n;i++)
            for(int j=1;j<=m;j++){
                f[i][j]=max(f[i-1][j],f[i][j-1]);
                f[i][j]=max(f[i][j],f[i-1][j-1]+nums1[i-1]*nums2[j-1]);
                 f[i][j]=max(f[i][j],nums1[i-1]*nums2[j-1]);
            }
       
        return f[n][m];
    }
};
```

#### 一维线性DP

##### [2944. 购买水果需要的最少金币数](https://leetcode.cn/problems/minimum-number-of-coins-for-fruits/)

> 思路：定义f[i]表示为获取前i个水果的最小金币数。通过选或者不选当前第i个水果来划分集合。如果不选。那么可以由前面的水果免费获得 即有 `f[i]=min{f[j-1]+cost[j]}`。同时必须要满足`j>=1&&2*j>=i`。如果选择，那么直接由上一个转移，即`f[i]=f[i-1]+cost[i]`。取最小值即可

```c++
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int minimumCoins(vector<int>& prices) {
        int n = prices.size();
        vector<int> dp(n + 1, INT_MAX);
        dp[0] = 0;

        for (int i = 1; i <= n; i++) {
            // 假设直接购买当前商品
            dp[i] = dp[i - 1] + prices[i - 1];

            // 从 j = i-1 开始遍历，查找可能的父节点
            for (int j = i - 1; j >= 1 && 2 * j  >= i; j--) {
                dp[i] = min(dp[i], dp[j-1] + prices[j-1]);
            }
        }
        return dp[n];
    }
};

```



### 区间DP问题

##### [516. 最长回文子序列](https://leetcode.cn/problems/longest-palindromic-subsequence/)

> 思路：区间DP的模板，定义`f[i][j]`表示所有i-j区间内的回文子序列长度，属性为`MAX`。集合可以划分为3个子集，分别是，选择左右端点s[i],s[j]，表示为`f[i+1][j-1]+2`。选择左端点不选右端点10，表示为`f[i][j-1]`，选择右端点不选择左端点，表示为`f[i+1][j]`。状态转移方程为 `f[i][j]=max(f[i][j-1],f[i+1][j],f[i+1][j-1]+2)`。注：选择左右端点情况不一定存在
>
> ### 关键点
>
> 1. 初始化时，单个字符的长度为1
> 2. **区间DP步骤：**
>    1. 枚举区间长度1-n
>    2. 枚举区间左端点，（右端点自动出来）
>    3. 状态计算。

```c++
class Solution {
public:
    int longestPalindromeSubseq(string s) {
        int n=s.size();
        vector<vector<int>> f(n+1,vector<int>(n+1,0));
		//单个字符是回文
        for(int i=1;i<=n;i++)
            f[i][i]=1;
        //枚举区间长度
        for(int len=2;len<=n;len++){
            for(int i=1;i+len-1<=n;i++){
                //枚举区间左端点
                int l=i,r=i+len-1;
                f[l][r]=max(f[l+1][r],f[l][r-1]);
                if(s[l-1]==s[r-1])
                f[l][r]=max(f[l][r],f[l+1][r-1]+2);
            }
        }
        return f[1][n];
    }
};
```

##### [:innocent: 730. 统计不同回文子序列](https://leetcode.cn/problems/count-different-palindromic-subsequences/)

> 说明：此题的思想很重要。对于去重问题的思路
>
> ### 思路：
>
> 与上题一致，同样属于区间DP的内容，定义f[i][j]为区间[i,j]内所有不同的非空回文子序列个数。属性为`sum`
>
> #### 集合划分：
>
> ##### 当 **s[i]==s[j]**
>
> 1. 需要进行计算内部与s[i]相同的字符个数
> 2. 如果字符个数等于0，那么说明不会有重复的序列，即`f[i][j]=2*f[i+1][j-1]+2`（原来的回文加上左右两个又产生一个新的回文，同时a，aa,也是两个回文）
> 3. 如果字符个数等于1，说明除了a不能算进去之外，其他都没有重复的序列，即`f[i][j]=2*f[i+1][j-1]+1`（内部的a与两端的a重复了，单个a仅能算一次）
> 4. 如果字符个数等于2，说明字符内部的子区间多算了一次，因此`f[i][j]=2*f[i+1][j-1]-f[left+1][right-1]`
>
> ##### 当**s[i]!=s[j]**
>
> 1. 状态转移方程为` f[i][j]=f[i+1][j]+f[i][j-1]-f[i+1][j-1]`（其中的[i+1,j]和[i,j-1]包含了两次[i+1,j-1]因此需要减去一次）

```c++
class Solution {
public:
    static const int MOD=1e9+7;
    int countPalindromicSubsequences(string s) {
        int n=s.size();
        vector<vector<long long>> f(n+1,vector<long long>(n+1,0));
        //单个字符是回文序列
        for(int i=1;i<=n;i++)
            f[i][i]=1;
        
        for(int len=2;len<=n;len++){
            //枚举起点
            for(int i=1;i+len-1<=n;i++){
                //大致分为四种情况
                /**
                 当s[l]==s[r]时，计算内部[l+1,r-1]与s[l]相等的字符个数
                 如果为0，那么不同的回文序列有 2*f[l+1][r-1]+2
                 如果为1  那么不同的回文序列有 2*f[l+1][r-1]+1;//单个字符已经被计算了
                 如果为2  那么不同的回文序列有 2*f[l+1][r-1]-f[left+1][right-1]
                 当s[l]!=s[r]
                 不同的回文序列有  s[l+1][r]+s[l][r-1]-s[l+1][r+1];//重复计算了一次s[l+1][r+1]; 
                **/
                int l=i,r=i+len-1;
                if(s[l-1]==s[r-1]){
                    //三种情况
                    //计算左边相同的位置
                    int left=l+1,right=r-1;
                    while(left<=right&&s[left-1]!=s[l-1])left++;
                    while(left<=right&&s[right-1]!=s[r-1])right--;
                   
                    if(left>right){
                        f[l][r]=2*f[l+1][r-1]+2;
                    }else if(left==right)
                        f[l][r]=2*f[l+1][r-1]+1;
                    else
                        f[l][r]=2*f[l+1][r-1]-f[left+1][right-1];
                }else
                    f[l][r]=f[l+1][r]+f[l][r-1]-f[l+1][r-1];

                f[l][r]=(f[l][r]+MOD)%MOD;
            }
        }
        return f[1][n];
    }
};
```

##### [1312. 让字符串成为回文串的最少插入次数](https://leetcode.cn/problems/minimum-insertion-steps-to-make-a-string-palindrome/)

>  思路：区间DP问题，定义f[i][j]表示为所有让区间i-j的串成为回文的操作次数，属性为`min`。集合划分为三个子集，当s[i]==s[j]时，不用进行插入，表示为`s[i+1][j-1]`。如果不相等，考虑在`i`前面插入，表示为`s[i+1][j]+1`。考虑在j后面插入,表示为`s[i][j-1]+1`
>
> 状态转移方程为 `f[i][j]=min(f[i+1][j]+1,f[i][j-1]+1,f[i+1][j-1])`注意：相等的情况不一定会出现。

```c++
class Solution {
public:

    int minInsertions(string s) {
        int n=s.size();
        vector<vector<int>> f(n+1,vector<int> (n+1,0));

        //初始化
        f[0][0]=0;
        for(int i=1;i<=n;i++)//只能在右边进行插入
            f[i][0]=i;
        for(int j=1;j<=n;j++)//只能在左边进行插入
            f[0][j]=j;
		//枚举区间长度
        for(int len=2;len<=n;len++){
            //枚举起点
            for(int i=1;i+len-1<=n;i++){
                //状态计算
                int l=i,r=i+len-1;
                f[l][r]=min(f[l+1][r],f[l][r-1])+1;
                if(s[l-1]==s[r-1])
                f[l][r]=min(f[l][r],f[l+1][r-1]);
            }
        }
        return f[1][n];
    }
};
```

##### [5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/)

> 思路：本题由于求得是子串，子串都是连续的，因此我们仅需要考虑内部的子串是否是回文即可，定义f[i][j]表示为区间[i,j]是否为回文串。集合可以划分为两个子集，当`s[i]==s[j]`时，如果有`f[i+1][j-1]==true`说明当前[i,j]是回文。表示为`f[i][j]=s[i][j]&&f[i+1][j-1]`当`s[i]!=s[j]`时，表示为`f[i][j]=false;`
>
> ### 关键点
>
> 1. 长度为1和长度为2的需要提前初始化，因此，当s[i]==s[j]时，我们需要判断他的内部是否为回文，因此长度要从3开始遍历

```c++
class Solution {
public:
    string longestPalindrome(string s) {
        int n = s.size();
        
        vector<vector<bool>> f(n + 1, vector<bool>(n + 1, false));
        int start = 1, maxLength = 1;
        
        // 初始化单个字符是回文
        for (int i = 1; i <= n; ++i) {
            f[i][i] = true;
        }
        
        // 初始化长度为2的回文子串
        for (int i = 1; i < n; ++i) {
            if (s[i - 1] == s[i]) {
                f[i][i + 1] = true;
                start = i;
                maxLength = 2;
            }
        }
        
        // 动态规划填充表格
        for (int len = 3; len <= n; ++len) {
            for (int i = 1; i + len - 1 <= n; ++i) {
                int l = i, r = i + len - 1;
                if (s[l - 1] == s[r - 1] && f[l + 1][r - 1]) {
                    f[l][r] = true;
                    start = l;
                    maxLength = len;
                }
            }
        }
        
        return s.substr(start - 1, maxLength);
    }
};

```

##### [3040. 相同分数的最大操作数目 II](https://leetcode.cn/problems/maximum-number-of-operations-with-the-same-score-ii/)

> 思路：此题依然是区间DP问题，主要是要使用三次不同的结果，定义**`f[i][j]`表示在区间i-j内子数组和为x的操作数**，属性为`max`。可以分为三种情况，在开头，在末尾，在开头末尾，开头表示为`f[i+2][j]+1`.在末尾表示为`f[i][j-2]+1`。在开头末尾表示为`f[i+1][j-1]+1`.状态转移方程为 `f[i][j]=max(f[i+1][j-1],f[i][j-2],f[i+2][j])+1`

```c++
class Solution {
public:
    static const int N = 2100;
    int f[N][N];

    int dfs(int n,int target, vector<int>& nums) {
        memset(f,0,sizeof f);
        for(int len=2;len<=n;len++){
            for(int i=1;i+len-1<=n;i++){
                int l=i,r=i+len-1;
                if(nums[l-1]+nums[l]==target)
                f[l][r]=max( f[l][r],f[l+2][r]+1);
                if(nums[l-1]+nums[r-1]==target)
                f[l][r]=max( f[l][r],f[l+1][r-1]+1);
                if(nums[r-1]+nums[r-2]==target)
                f[l][r]=max( f[l][r],f[l][r-2]+1);
            }
        }

        return f[1][n];
    }

    int maxOperations(vector<int>& nums) {
        int n = nums.size();
        memset(f, -1, sizeof(f)); // 初始化记忆化数组为-1

        if (n < 2) return 0; // 如果数组长度小于2，直接返回0

        int sum1 = nums[0] + nums[1]; // 前两个元素的和
        int sum2 = nums[0] + nums[n - 1]; // 第一个和最后一个元素的和
        int sum3 = nums[n - 2] + nums[n - 1]; // 后两个元素的和

        // 分别计算每种情况的最大操作次数
        int res1 = dfs(n,sum1, nums);
        int res2 = dfs(n,sum2, nums);
        int res3 = dfs(n,sum3, nums);

        // 返回最大值
        return max({res1, res2, res3});
    }
};

```



### 状态压缩DP

#### 排列型

##### [526. 优美的排列](https://leetcode.cn/problems/beautiful-arrangement/)

> 思路：此题能够看出是一个全排列的问题，每次必须选择一个数，也就是说，最终合法的数组长度为n，同时，每个整数仅能选择一次，如果采用回溯做法，时间复杂度为`n!`过于复杂。我们通过`1-n`中的数字采用一个状态来进行压缩，`0`表示位选，`1`表示选择。对于每个状态s，我们可以利用`(s>>i)&1==1`来进行判断当前**第`i`个数字**是否已被选择。因此可以节省时间，时间复杂度能优化到`O（2^n*n)`。
>
> 具体思路：定义`f[i][s]`表示构造长度为`i`，选择状态为`s`的方案个数。由于每次必须选择，因此仅有一种情况，必须选，那么当前的状态`f[i][s]`的值相当于所有`f[i-1][s^(1<<(j-1))]`的总和。
>
> ### 关键点
>
> 1. 对于每个状态，需要保证当前枚举的`f[i][s]`中的`i`已被选择同时满足`（i%j==0||j%i==0）`。才能够去加上上一个状态

```c++
class Solution {
public:
    int countArrangement(int n) {
        int mask=1<<n;
        vector<vector<int>> f(n+1,vector<int>(mask,0));
        f[0][0]=1;
        //遍历挑选个数
        for(int i=1;i<=n;i++){
            //枚举状态
            for(int s=0;s<mask;s++){
                //枚举数字
                for(int j=1;j<=n;j++){
                    if((s>>j-1)&1==1&&(i%j==0||j%i==0))
                    f[i][s]+=f[i-1][s^(1<<j-1)];
                }
            }
        }
        return f[n][mask-1];
    }
};
```

##### [1879. 两个数组最小的异或值之和](https://leetcode.cn/problems/minimum-xor-sum-of-two-arrays/)

> 思路：如果采用回溯做法时间复杂度很高，同时下方提示告诉我们n<=14，考虑使用状态压缩DP，定义`f[i][s]`表示`nums1`中选择前`i`个同时在`nums2`中选择状态为`s`的异或值之和。属性为`min`。`s`要表示`n`位数的状态，因此需要设置`s`最大为`2^n`。必选选择，因此只有一种情况，即表示为如果当前枚举nums2中的第j个数已经被选中，那么`f[i][s]=min(f[i-1][s^(1<<(j-1))]+(nums1[i-1]^nums2[j-1]))`其中==1<=j<=n==。
>
> ### 关键点
>
> 1. 为什么是要往移动`j-1`呢？这是因为，二进制中我们的下标是从0开始的，但是我们从1开始遍历，那么二进制第一个位置的值，就只需要移动`0`位就可以了。因此需要`j-1`
> 2. 返回值，我们返回在前`n`个数中选择，同时`nums2`状态标识为全部选中（即`mask-1`）。返回`f[n][mask-1]`即可
> 3. 统计集合中元素的个数:使用`lowbit`

```c++
class Solution {
public:
    const int INF=0x3f3f3f3f;
    //使用lowbit计算集合中元素的个数
    int count(int s){
        int cnt=0;
        for(int t=s;t;t-=t&(-t))
            cnt++;
        return cnt;
    }
    int minimumXORSum(vector<int>& nums1, vector<int>& nums2) {
        int n=nums1.size();
        int mask=1<<n;
        vector<vector<int>> f(n+1,vector<int>(mask,INF));
        f[0][0]=0;
        //遍历当前选择到了第几个数
        for(int i=1;i<=n;i++){
            for(int s=0;s<mask;s++){
                if(count(s)!=i)continue;
                for(int j=1;j<=n;j++){
                    if(((s>>(j-1))&1)==0)continue;//集合中不包含当前元素
                    f[i][s]=min(f[i][s],f[i-1][s^(1<<(j-1))]+(nums1[i-1]^nums2[j-1]));
                }
            }
        }
        return f[n][mask-1];
    }
};
```

##### [996. 平方数组的数目](https://leetcode.cn/problems/number-of-squareful-arrays/)

>  思路：要求排列数量，第一反应暴搜+剪枝。能过，不过要注意去重的问题，对于同一层的分支，如果值相同的话，需要进行去重，去重操作：1.对数组排序。2.同一层分支（for循环内）进行去重`if(i!=0&&nums[i]==nums[i-1]&&!st[i-1]) continue`
>
> ### 关键点
>
> 1. 使用bool 类型的数组st 来记录元素是否已被访问。
> 2. 去重操作。
>
> 思路2：采用状态压缩DP，定义`f[s][i]`表示为状态为s，同时最后一次选择第i个元素的方案数。集合由n个子集构成，我们以倒数第二次选择的第j个数来划分子集。可以划分为，1,2,3,…..n。不包括i。假设倒数第二次选择第k个数，表示为`f[i][s]+=f[s\{i}][k]`。`1<=k<=n`即不包含i，同时最后一次选择为k。最后遍历加上所有的`f[mask-1][i]`即可
>
> ### 注：
>
> 思路2：去重不会

```c++
//思路1
class Solution {
public:
    int cnt = 0;
    vector<int> res;
    bool st[14] = {false};  // 状态数组，用于标记数字是否被使用

    bool isPerfectSquare(int num) {
        int root = static_cast<int>(sqrt(num));
        return root * root == num;
    }

    void dfs(int u, int n, vector<int>& nums) {
        if (u == n) {
            cnt++;
            return;
        }

        for (int i = 0; i < n; i++) {
            if (st[i]) continue;  // 如果该数字已经被使用，跳过
            if (!res.empty() && !isPerfectSquare(nums[i] + res.back())) continue;  // 剪枝，如果与上一个数之和不是完全平方数，跳过
            if(i!=0&&nums[i]==nums[i-1]&&!st[i-1]) continue;
            res.push_back(nums[i]);  // 将当前数字加入结果集
            st[i] = true;  // 标记当前数字已被使用
            dfs(u + 1, n, nums);  // 递归调用
            st[i] = false;  // 恢复状态
            res.pop_back();  // 移除当前数字
        }
    }

    int numSquarefulPerms(vector<int>& nums) {
        //res.reserve(nums.size());
        sort(nums.begin(),nums.end());
        dfs(0, nums.size(), nums);
        return cnt;
    }
};


//思路2（未去重）
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

class Solution {
public:
    bool isPerfectSquare(int num) {
        int root = static_cast<int>(sqrt(num));
        return root * root == num;
    }

    int numSquarefulPerms(vector<int>& nums) {
        int n = nums.size();
        int mask = 1 << n;
        vector<vector<int>> f(mask, vector<int>(n+1, 0));
        
        // 初始状态，选择一个数
        for (int i = 1; i <= n; ++i) {
            f[1 << (i - 1)][i] = 1;
        }

        for (int s = 1; s < mask; ++s) {
            for (int i = 1; i <= n; ++i) {
                // 如果当前数不在集合中，不可能是最后一次选择的数
                if (!(s & (1 << (i - 1)))) continue;

                for (int j = 1; j <= n; ++j) {
                    if (i == j || !(s & (1 << (j - 1)))) continue;

                    int sum = nums[i - 1] + nums[j - 1];
                    if (!isPerfectSquare(sum)) continue;

                    f[s][i] = (f[s][i] + f[s ^ (1 << (i - 1))][j]) % 1000000007;
                }
            }
        }

        int cnt = 0;
        for (int i = 1; i <= n; ++i) {
            cnt = (cnt + f[mask - 1][i]);
        }
		//待去重
        return cnt;
    }
};

```

##### [2741. 特别的排列](https://leetcode.cn/problems/special-permutations/)

> 思路：采用状态压缩DP，定义`f[s][i]`表示状态为s，最后一次选择第i个数的方案个数。按照倒数第二次选择第j个数来划分子集，可以划分为1,2,3,4,5…n。如果满足条件，那么即有，`f[s][i]+=f[s\{i}][j]``1<=j<=n`。最后遍历最后一层`f[mask-1][i]`
>
> ### 关键点
>
> 1. 如果状态为s，当前最后选择为i，但是状态中不包含i，那么表示不合法，跳过此次循环
> 2. 如果划分子集时，遍历j，如果j不在集合中，说明不合法，因此j是倒数第二次选择的，一定在集合中。

```c++
class Solution {
public:
    static const int MOD = 1e9 + 7;
    int specialPerm(vector<int>& nums) {
        int n = nums.size();
        vector<vector<int>> f(1 << n, vector<int>(n+1, 0));//定义为选择状态为s，最后一次选择第i个数
        for(int i=1;i<=n;i++){
            f[1<<(i-1)][i]=1;
        }

        //枚举状态
        for(int s=1;s<(1<<n);s++){
            //枚举最后一次选择第i个数
            for(int i=1;i<=n;i++){
                //如果说s里面已经不包括当前位置数,剪枝
                if(!(s>>(i-1)&1))continue;
                //枚举上一次要选择的数
                for(int j=1;j<=n;j++){
                    if(i==j||!(s>>(j-1)&1))continue;//如果选择了第i个数，第j个数还没有选，那么也不符合条件
                    if(nums[i-1]%nums[j-1]!=0&&nums[j-1]%nums[i-1]!=0)continue;
                    f[s][i]=(f[s][i]+f[s^(1<<(i-1))][j])%MOD;
                }
            }
        }
       int res = 0;
        for (int i = 1; i <=n; i++) {
            res = (res + f[(1 << n) - 1][i]) % MOD;
        }
        return res;
    }
};
```

#### 子集型

##### [2305. 公平分发饼干](https://leetcode.cn/problems/fair-distribution-of-cookies/)

> 思路：此题为子集型的问题，如果考虑回溯的做法，那么即枚举所有的自己，选择这个分支最大值中，最小的一个即可。需要使用一个数组来记录每个小朋友在当前分支中得到的饼干总和。同时可以进行剪枝优化。进行同一层分支的优化，如果当前不是第一个，并且当前分配的饼干跟前一个一样，说明是重复的结果，可以进行剪枝优化。最后统计处最小的最大值。
>
> 思路2：采用状态压缩DP，定义`f[i][s]`表示为前`i`个小朋友在分配状态为s下的不公平程度，属性为`Min`。集合可以通过枚举当前第`i`个小朋友可以选择的饼干状态进行枚举，如果当前第`i`个小朋友的分配状态为`p`，前`i-1`个小朋友的分配状态为`s\{p}`那么前`i-1`个小朋友的最小化不公平程度为`f[i-1][s^p]`。即`f[i][s]=min(f[i][s],max(f[i-1][s^p],sum[p]))`。我们需要使用一个数组来记录不同的分配状态下所能够得到的饼干总和。也就是说，当分配状态为`p`时，能够获得的饼干数量是多少
>
> ### 关键点
>
> 1. 使用`sum`数组来记录不同状态的饼干数量

```c++
//回溯
#include <algorithm>
#include <climits>
#include <vector>
using namespace std;

class Solution {
public:
    vector<int> child;
    //dfs(i)代表当前饼干cookies[i]要分给1-k哪个孩子
    int res=INT_MAX;
    void dfs(int i,int k,vector<int> &cookies){
        if(i==cookies.size()){
            //饼干已经分完了
            res=min(res,*max_element(child.begin(),child.end()));
            return ;
        }

        //分配饼干
        for(int j=0;j<k;j++){
           if (j > 0 && child[j]==child[j - 1] ) continue;
            child[j]+=cookies[i];
            dfs(i+1,k,cookies);
            child[j]-=cookies[i];
        }
    }
    int distributeCookies(vector<int> &cookies, int k) {
        child.resize(k+1,0);
        dfs(0,k,cookies);
        return res;
    }
};

//状态压缩DP
class Solution {
public:
    const int INF=0x3f3f3f3f;
    //使用状态压缩DP
    int distributeCookies(vector<int>& cookies, int k) {
        int n=cookies.size();
        int mask=1<<n;
        vector<int> sum(mask,0);//用于存储每个分配状态对应的不公平程度
        vector<vector<int>> f(k+1,vector<int>(mask,INF));
        //存储每个状态对应的不公平程度
        for(int s=0;s<mask;s++){
           for(int j=0;j<n;j++){
                if(s&(1<<j))
                sum[s]+=cookies[j];
           }
        }
        f[0][0]=0;
        //枚举每个小朋友
        for(int i=1;i<=k;i++){
            for(int s=0;s<mask;s++){
                for(int p=s;p;p=s&(p-1)){
                    f[i][s]=min(f[i][s],max(f[i-1][s^p],sum[p]));
                }
            }
        }
        return f[k][mask-1];
    }
};
```

##### [1723. 完成所有工作的最短时间](https://leetcode.cn/problems/find-minimum-time-to-finish-all-jobs/)

> 思路：与上题类似

```c++
class Solution {
public:
    const int INF=0x3f3f3f3f;
    int minimumTimeRequired(vector<int>& jobs, int k) {
        int n=jobs.size();
        int mask=1<<n;
        vector<int> sum(mask,0);
        vector<vector<int>> f(k+1,vector<int>(mask,INF));
        //初始化不同状态对应的工作时间
        for(int p=0;p<mask;p++){
            for(int j=0;j<n;j++){
                if(p&(1<<j))
                sum[p]+=jobs[j];
            }
        }
        f[0][0]=0;
        for(int i=1;i<=k;i++){
            for(int s=0;s<mask;s++){
                //枚举所有可能选择的子集状态
                for(int p=s;p;p=s&(p-1)){
                    f[i][s]=min(f[i][s],max(f[i-1][s^p],sum[p]));
                }
            }
        }
        return f[k][mask-1];
    }
};
```

##### [1986. 完成任务的最少工作时间段](https://leetcode.cn/problems/minimum-number-of-work-sessions-to-finish-the-tasks/)

> 思路：采用回溯，如果我们使用普通的回溯进行枚举，会超时，我们可以创建一个数组为`times[],`其中`times[i]`代表第i个时间段的耗时，当前工作`tasks[u]`可以分为两种情况，1.优先选择前面能够塞下的时间段进行塞下2.都塞不下了，只能新开一个时间段。
>
> 思路2：采用状态压缩DP，定义`f[s]`完成状态为`s`的任务，需要的最小时间段，集合可以通过倒数第二次完成的任务来划分，也就是当前状态`s`的子集。如果当前完成的任务为`p`，那么上一次完成的后的状态为`s\{p}`，则可以表示为`f[s]=min(f[s],f[s^p]+1)`注意：`f[s^p]`不一定存在，有可能当前的子集耗时超过了最大时间。
>
> ### 关键点
>
> 1. 进行状态的定义：此题为完成所有任务的最小时间段，就定义为完成状态为s的最小时间段
> 2. 子集的枚举
> 3. 可以预先处理好每个状态的耗时。

```c++
//回溯的做法
class Solution {
public:
    vector<int> times;//表示第i个时间段耗时多少
    int res=0x3f3f3f3f;
    void dfs(int u,int k,vector<int> &tasks,int sessionTime){
        if(u==tasks.size()){
            res=min(res,k);
            return ;
        }

        if(k>=res)return;
        //尽可能的完成
        for(int i=0;i<k;i++){
            //如果前面的时间段加上当前需要的时间满足要求
            if(times[i]+tasks[u]<=sessionTime){
                times[i]+=tasks[u];
                dfs(u+1,k,tasks,sessionTime);
                times[i]-=tasks[u];
            }
        }
        //无法赛进去
        times[k]=tasks[u];
        dfs(u+1,k+1,tasks,sessionTime);
        times[k]=0;
    }
    int minSessions(vector<int>& tasks, int sessionTime) {
        int n=tasks.size();
        times.resize(20,0);
        dfs(0,0,tasks,sessionTime);
        return res;
    }
};


//状态压缩DP
class Solution {
public:
    const int INF=0x3f3f3f3f;
    int minSessions(vector<int>& tasks, int sessionTime) {
        int n = tasks.size();
        int mask = 1 << n;
        vector<int> sum(mask, 0);
        vector<int> f(mask,INF);

        // 计算每个子集的任务总和
        for (int s = 0; s < mask; s++) {
            for (int j = 0; j < n; j++) {
                if (s & (1 << j)) {
                    sum[s] += tasks[j];
                }
            }
        }
        f[0]=0;
        // 初始化单个任务的状态
        for (int i = 0; i < n; i++) {
            f[1 << i] = 1;
        }

        for(int s=0;s<mask;s++){
            for(int p=s;p;p=s&(p-1)){
                if(sum[p]>sessionTime)continue;
                f[s]=min(f[s],f[s^p]+1);
            }
        }

        // 找到最终结果
        // int res = INT_MAX;
        // for (int i = 0; i < n; i++) {
        //     res = min(res, f[mask - 1][i]);
        // }

        return f[mask-1];
    }
};
```

##### [698. 划分为k个相等的子集](https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/)

> 思路：按照回溯的思想，我们需要装k个子集，同时每个子集的和为`sum/k`，sum为数组的总和，如果sum不能被k整除`，即sum%k!=0`。此时说明不能划分为k个子集，直接返回false，否则的话可以尝试。使用回溯，定义一个数组f,代表每个子集的和，dfs(u)代表当前元素应该选择哪个子集进行放入。如果最终所有的元素都能够放下，说明能够划分成功，否则不能。
>
> ### 关键点
>
> 1. 剪枝优化1，可以对数组进行降序排序，让大的先被选，这样可以减少排列树的复杂度。进行一个去重操作，如果前一个桶的元素和跟当前桶的元素和相同，说明当前元素放在当前桶跟放在上一个桶情况一样，发生了重复，进行重复性剪枝。`if(i&&f[i]==f[i-1])continue;`
> 2. 剪枝优化2，如果当前的桶不能放下元素，那么进行剪枝即可。`if(f[i]+nums[u]>target)continue;`

##### [473. 火柴拼正方形](https://leetcode.cn/problems/matchsticks-to-square/)

> 思路：利用回溯来完成，我们有两种视角，第一种，球选择桶，按照题意，我们一共需要装满4个桶，因此，dfs(u)表示当前第u个球，会选择哪一个桶进行装入。最后如果全部的球都装进了4个桶，说明能够成功，否则不能。
>
> 视角2：桶选择球，dfs(u)代表当前第u个桶，会选择哪些球进行装入，需要设置一个bool数组 used 表示每个球的使用情况。如果每个桶都装满了，说明能够成功，否则不能。
>
> 思路2：利用状态压缩，定义f[s]表示当前选择状态为s，正方形未放满的一条边长度，`f[0]=0`,其余状态初始化为`-1`，代表不可达。以最后一次挑选的火柴来划分集合，可以划分为1,2,3,4,….n,如果当前最后一次挑选火柴放入集合的长度小于`tar`说明可以放入，更新`f[s]=(f[s^(1<<j)+nums[j]])%tar`。最终如果所有的火柴放入后的长度为0，说明能够构成正方形。

```c++
//思路1
class Solution {
public:
    static const int N=16;
    //bool used[N];
    vector<int> cursum;
    bool dfs(int u,vector<int> &nums,int tar){
        if(u==nums.size())return true;
        //挑选适合的火柴加入桶中
        for(int i=0;i<4;i++){
            if(i&&cursum[i]==cursum[i-1])continue;
            if(cursum[i]+nums[u]>tar)continue;
            cursum[i]+=nums[u];
            if(dfs(u+1,nums,tar))return true;
            cursum[i]-=nums[u];
        }
        return false;
    }
    bool makesquare(vector<int>& nums) {
        cursum.resize(5,0);
        int sum=accumulate(nums.begin(),nums.end(),0);
        sort(nums.rbegin(),nums.rend());//倒序排
        if(sum%4!=0)return false;
        int tar=sum/4;
        return dfs(0,nums,tar);

    }
};


//视角2
class Solution {
public:
    static const int N=16;
    bool used[N];
    bool dfs(int u,int k,int tar,int cursum,vector<int> &nums){
        if(k==0)return true;
        if(cursum==tar)
        return dfs(0,k-1,tar,0,nums);
       //选择球
       for(int i=u;i<nums.size();i++){
            if(used[i]||cursum+nums[i]>tar)continue;//剪枝
            used[i]=true;
            if(dfs(i+1,k,tar,cursum+nums[i],nums))return true;
            used[i]=false;
            if(cursum==0)return false;
       }
        return false;
    }
    bool makesquare(vector<int>& nums) {
        int sum=accumulate(nums.begin(),nums.end(),0);
        if(sum%4!=0)return false;
        int tar=sum/4;
         if(nums[0]>tar)return false;
        return dfs(0,4,tar,0,nums);
    }
};

//class Solution {
public:
    bool makesquare(vector<int>& nums) {
        int sum=accumulate(nums.begin(),nums.end(),0);
        if(sum%4!=0)return false;
        int tar=sum/4;
        int n=nums.size();
        int mask=1<<n;
        vector<int> f(mask,-1);

        f[0]=0;
        //枚举状态
        for(int s=0;s<mask;s++){
            //枚举火柴
            for(int j=1;j<=n;j++){
                //如果当前集合中不包含火柴，跳过
                if(((s>>(j-1))&1)==0)continue;
                //如果当前集合去掉这个火柴，同时长度小于等于tar说明当前火柴可以去掉，因此长度变为取余
                if(f[s^(1<<(j-1))]>=0&&f[s^(1<<(j-1))]+nums[j-1]<=tar)
                f[s]=(f[s^(1<<(j-1))]+nums[j-1])%tar;
            }
        }
        return f[mask-1]==0;
    }
};
```



##### [2002. 两个回文子序列长度的最大乘积](https://leetcode.cn/problems/maximum-product-of-the-length-of-two-palindromic-subsequences/)

> 思路：回溯，利用回溯枚举所有可能的串，单个字符有三种情况，不选择该字符，a串选择，b串选择。因此利用回溯枚举所有的情况即可。
>
> 思路2：状态压缩+求最长回文长度。枚举所有状态s，同时枚举当前字符应该加入哪个串，如果当前状态包含j，那么加入a串，如果不包含，加入b串。每一次都枚举到末尾。最后分别计算a串的回文长度以及b串的回文长度，更新`res`

```c++
class Solution {
public:
    vector<char> str1,str2;
    int res=0;
    void dfs(int i,int len,string &s){
        if(i==len){
            //判断一下是否符合规则
            int n1=str1.size();
            int n2=str2.size();
            if(res<n1*n2&&ishuiwen(str1)&&ishuiwen(str2))
                res=n1*n2;
            return ;
        }
        //不选择该字符
        dfs(i+1,len,s);

        //添加到a中
        str1.emplace_back(s[i]);
        dfs(i+1,len,s);
        str1.pop_back();

        //添加到b中

        str2.emplace_back(s[i]);
        dfs(i+1,len,s);
        str2.pop_back();
    }
    bool  ishuiwen(vector<char> &str){
        int n=str.size();
        for(int i=0,j=n-1;i<j;i++,j--){
            if(str[i]!=str[j])
                return false;
        }
        return true;
    }
    int maxProduct(string s) {
        dfs(0,s.size(),s);
        return res;
    }
};


//状态压缩
class Solution {
public:
    int lcs(string& s) {
        int n = s.size();
        vector<vector<int>> f(n + 2, vector<int>(n + 2, 0)); // 初始化为1
        for(int i=1;i<=n;i++)
            f[i][i]=1;
        // 遍历所有可能的子串长度
        for (int len = 2; len <= n; len++) {
            // 枚举左端点
            for (int i = 1; i + len - 1 <= n; i++) {
                int l = i, r = i + len - 1;
                f[l][r] =
                    max(f[l][r - 1], f[l + 1][r]); // 去掉一个字符后的最大值
                if (s[l - 1] == s[r - 1]) {
                    f[l][r] = max(f[l][r], f[l + 1][r - 1] + 2); // 匹配的字符增加2    
                }
            }
        }
        return f[1][n];
    }
    int maxProduct(string str) {
        int res=0;
        //枚举所有的状态
        int n=str.size();
        int mask=1<<n;
        for(int s=0;s<mask;s++){
            string a="",b="";
            //枚举存在的子串
            for(int j=1;j<=n;j++){
                if((s>>(j-1)&1)!=0)
                    a+=str[j-1];
                else
                    b+=str[j-1];
            }
            if(a.empty()||b.empty())continue;
            //更新结果
            int lena=lcs(a);
            int lenb=lcs(b);
            res=max(res,lena*lenb);
        }
        return res;
    }
};
```

### 划分型DP问题

#### 判断是否存在



##### [2369. 检查数组是否存在有效划分](https://leetcode.cn/problems/check-if-there-is-a-valid-partition-for-the-array/)

> 思路：定义f[i]表示为前i个数字能否有效的划分，由最后一次划分来划分集合。集合可以分为，最后一次划分2个数，3个数。
>
> 状态表示为`f[i]=f[i]||f[i-2]`此时的必须满足条件，最后一次划分成立。以及`f[i]=f[i]||f[i-3]`。表示最后一次划分有三个数字。
>
> ### 关键点
>
> 1. 只有前一个状态成立，才能够进行划分下一个状态`。if (f[i-2]&&nums[i - 1] == nums[i - 2]) `简写为`if(nums[i-1]==nums[i-2])f[i]=f[i]||f[i-2]`

```c++
class Solution {
public:
    bool validPartition(vector<int>& nums) {
        int n = nums.size();
        vector<bool> f(n + 1, false); // 定义 f[i] 为前 i 个数是否是有效划分。
        f[0] = true;

        for (int i = 2; i <= n; i++) {
            if (nums[i - 1] == nums[i - 2]) {
                f[i] = f[i] || f[i - 2]; // 检查两个相等的数
            }
            if (i >= 3) {
                if (nums[i - 1] == nums[i - 2] && nums[i - 2] == nums[i - 3]) {
                    f[i] = f[i] || f[i - 3]; // 检查三个相等的数
                }
                if (nums[i - 1] - nums[i - 2] == 1 && nums[i - 2] - nums[i - 3] == 1) {
                    f[i] = f[i] || f[i - 3]; // 检查三个递增的数
                }
            }
        }
        return f[n];
    }
};

```

##### [139. 单词拆分](https://leetcode.cn/problems/word-break/)

> 思路：定义`f[i]`表示为前i个字符能否由字典中的字符串构成，由最后一次划分来分割集合，最后一次划分可以选择第一个字符串，第2个，第3个….第k个.。因此状态表示为`f[i]=f[i]||f[i-len]`。（表示为上一次的划分必须成立才可以）
>
> ### 关键点
>
> 1. 集合划分成k个，因此要做一个遍历
> 2. 最后一次选择的字符串跟s中的`i-len,i`进行对比

```c++
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        int n = s.size();
        int m = wordDict.size();
        vector<bool> f(n + 1, false);
        f[0] = true; // 空字符串是可以构成的

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                int len = wordDict[j-1].size();
                if (i < len) continue;
                // 判断从 s[i-len:i] 是否与字典中的某个单词匹配
                if (s.substr(i - len, len) == wordDict[j-1]) {
                    f[i] = f[i] || f[i - len];
                }
            }
        }
        return f[n];
    }
};
```

#### 求最大最小

##### [132. 分割回文串 II](https://leetcode.cn/problems/palindrome-partitioning-ii/)

> 思路：定义f[i]表示为将1-i分割成回文串的次数，属性为`min`，使用最后一次的子数组来划分集合，枚举子数组的左端点L，考虑`f[L]`转移到`f[i]`，同时考虑`s[L,i]`是否符合条件。根据题目意思划分子集的长度为1,2，…..i。因此表示为`f[i]=min(f[i],f[i-len]+1)`。注意：此时只有当前`s[L,i]`是回文串是才能进行转移条件。
>
> ### 关键点
>
> 1. 判断一个区间的字符串是否为回文串，我们可以使用DP数组进行预处理，区间DP即可。
> 2. 对于`1-i`的串，如果是回文串，那么切割次数为0.

```c++
class Solution {
public:
    vector<vector<bool>> st;

    // 判断一个字符串是否是回文
    void judge(const string &s) {
        int n = s.size();
        // 初始化单个字符
        st.assign(n, vector<bool>(n, false));
        for (int i = 0; i < n; ++i) {
            st[i][i] = true;
        }
        // 初始化两个字符
        for (int i = 0; i < n - 1; ++i) {
            if (s[i] == s[i + 1]) {
                st[i][i + 1] = true;
            }
        }
        // 枚举长度
        for (int len = 3; len <= n; ++len) {
            for (int i = 0; i + len - 1 < n; ++i) {
                int l = i, r = i + len - 1;
                if (s[l] == s[r] && st[l + 1][r - 1]) {
                    st[l][r] = true;
                }
            }
        }
    }

    int minCut(string s) {
        int n = s.size();
        st.assign(n, vector<bool>(n, false));
        judge(s);
        
        vector<int> f(n + 1, 0x3f3f3f3f);
        
        
        for (int i = 1; i <= n; ++i) {
            //本身为回文串不需要进行切割
            if(st[0][i-1]){
                f[i]=0;
                continue;
            }
            for (int j = 1; j <= i; ++j) {
                if (st[j - 1][i - 1]) { // j-1到i-1是回文
                    f[i] = min(f[i], f[j - 1] + 1);
                }
            }
        }
        
        return f[n] ; // 返回最小切割数，实际切割次数比分割的块数少1
    }
};

```

##### [2707. 字符串中的额外字符](https://leetcode.cn/problems/extra-characters-in-a-string/)

> 思路1：反向思考，定义`f[i]`表示为`1-i`中在字典中存在的字符个数，属性为`max`，以最后一次的子串来划分集合，枚举子串长度，左端点为L，考虑从`f[L]`转移到`f[i]`。只有当子串在字典中才进行转移。表示为`f[i]=max(f[i],f[i-len]+len)`。最后返回`n-f[n]`即为最小的额外字符个数
>
> 思路2：定义`f[i]`表示为`1-i`中额外字符的最小个数，以最后一次的子串来划分集合，枚举子串长度，左端点为`L`，考虑从`f[L]`转移到`f[i]`。只有当子串**不**在字典中才进行转移，表示为`f[i]=min(f[i],f[L])`

```c++
class Solution {
public:
    int minExtraChar(string s, vector<string>& dictionary) {
        int n=s.size();
        vector<int> f(n+1,0);
        unordered_set<string> set;
        for(auto str:dictionary)
        set.insert(str);
        for(int i=1;i<=n;i++){
            //枚举区间长度
            f[i]=f[i-1];
            for(int j=1;j<=i;j++){
                int l=i-j,r=i;
                string temp=s.substr(l,j);
                if(set.find(temp)==set.end())continue;
                f[i]=max(f[i],f[i-j]+j);
            }
        }
        

        return n-f[n];
    }
};


class Solution {
public:
    int minExtraChar(string s, vector<string>& dictionary) {
        unordered_set<string>  set(dictionary.begin(),dictionary.end());
        int n=s.size();
        vector<int> f(n+1,0x3f3f3f3f);
        f[0]=0;
        for(int i=1;i<=n;i++){
            f[i]=f[i-1]+1;
            //枚举区间长度
            for(int len=1;len<=i;len++){
                int l=i-len;
                string temp=s.substr(l,len);
                if(set.find(temp)==set.end())continue;
                f[i]=min(f[i],f[l]);
            }
        }
        return f[n];
    }
};
```

##### [2767. 将字符串分割为最少的美丽子字符串](https://leetcode.cn/problems/partition-string-into-minimum-beautiful-substrings/)

> 思路：采用划分型DP，定义f[i]表示为前i个字符能够划分成子字符串的个数，属性为`min`。以最后一个子字符串的长度来进行划分集合。长度可以是1,2,3，……i，不能超过当前的i，即可表示为`f[i]=min(f[i-len]+1)`。前提是必须符合条件，同时`1<=len<=i`。
>
> ### 关键点
>
> 1. 由于是求最大和最小，因此f需要初始化为`INF`
> 2. f[0]=0,因为空字符串能够划分成的字符串个数为0.

```c++
class Solution {
public:
    // 判断一个字符串是否表示一个5的幂
    bool isPowerOfFive(const string& s) {
        long long num = 0;
        for(char c : s) {
            num = num * 2 + (c - '0');
            // 如果数字超过了5的最大幂次，直接返回false
            if(num > 1220703125) return false;
        }
        // 判断这个数字是否是5的幂
        while(num > 1 && num % 5 == 0) {
            num /= 5;
        }
        return num == 1;
    }

    int minimumBeautifulSubstrings(string s) {
        int n = s.size();
        vector<int> f(n+1, 0x3f3f3f3f);
        f[0] = 0;
        // 枚举当前到了第i个字符
        for(int i = 1; i <= n; i++) {
        //枚举最后一个子字符串的长度
            for(int len = 1; len <= i; len++) {
                int l = i - len, r = i;
                string temp = s.substr(l, len);
                if(temp[0] == '0') continue;//判断是否有前导0
                if(isPowerOfFive(temp)) {//判断是否是5的幂
                    f[i] = min(f[i], f[l] + 1);
                }
            }
        }
        return f[n] == 0x3f3f3f3f ? -1 : f[n];
    }
};

```

##### [91. 解码方法](https://leetcode.cn/problems/decode-ways/)

> 思路：采用划分型DP，定义`f[i]`表示为前i个字符的解码方式个数，以最后一次的子字符串来划分集合。由题可知，子字符串的长度仅能是1或者2，因此遍历长度仅需要遍历两次即可。如果有前导0，说明不符合，同时，如果转换成数字，大小超过26或者小于1也不符合。表示为`f[i]+=f[i-len]`
>
> ### 关键点
>
> 1. 以最后一次的子字符串来划分集合。
> 2. 子字符串长度小于等于2
> 3. **空字符串可以有 1 种解码方法，解码出一个空字符串**

```c++
class Solution {
public:
    int numDecodings(string s) {
        int n=s.size();
        vector<int> f(n+1,0);
        //初始化
        f[0]=1;//空字符串有一种解码方式
        for(int i=1;i<=n;i++){
            for(int len=1;len<=2;len++){
                int l=i-len,r=i;
                if(l<0)continue;
                if(s[l]=='0')continue;//前导0
                string temp=s.substr(l,len);
                int num=stoi(temp);
                if(num<=0||num>26)continue;
                f[i]=f[i]+f[l];
            }
        }
        return f[n];
    }
};
```

##### [639. 解码方法 II](https://leetcode.cn/problems/decode-ways-ii/)

> 思路：同样是划分型DP，定义f[i]表示前i个字符的解码方式个数。以最后一次的字符串长度来划分集合，有两种情况，分别是长度为1，长度为2，的情况，长度为1，需要考虑是否是`*`，如果是那么表示为f[i]+=f[i-len]*9，如果不是，那么即为f[i]+=f[l]。如果长度为2，需要考虑4中情况，分别是`x*`,`*x`,`**`,`xx`。分类讨论即可

```c++
class Solution {
public:
    static const int MOD = 1e9 + 7;

    int numDecodings(string s) {
        int n = s.size();
        vector<long long> f(n + 1, 0);
        f[0] = 1; // 空字符串构成的解码方式有1种

        for (int i = 1; i <= n; i++) {
            for (int len = 1; len <= 2; len++) {
                int l = i - len;
                if (l < 0) continue;

                if (len == 1) {
                    if (s[l] == '*') {
                        f[i] = (f[i] + f[l] * 9) % MOD;
                    } else if (s[l] != '0') {
                        f[i] = (f[i] + f[l]) % MOD;
                    }
                } else {
                    if (s[l] == '0') continue;

                    if (s[l] == '*' && s[l + 1] == '*') {
                        f[i] = (f[i] + f[l] * 15) % MOD; // '**' 可以表示 '11'-'19' 和 '21'-'26'
                    } else if (s[l] == '*') {
                        if (s[l + 1] <= '6') {
                            f[i] = (f[i] + f[l] * 2) % MOD; // '1*'-'6*' 可以表示 '10'-'16' 或 '20'-'26'
                        } else {
                            f[i] = (f[i] + f[l]) % MOD; // '7*'-'9*' 只能表示 '17'-'19'
                        }
                    } else if (s[l + 1] == '*') {
                        if (s[l] == '1') {
                            f[i] = (f[i] + f[l] * 9) % MOD; // '1*' 可以表示 '11'-'19'
                        } else if (s[l] == '2') {
                            f[i] = (f[i] + f[l] * 6) % MOD; // '2*' 可以表示 '21'-'26'
                        }
                    } else {
                        string temp = s.substr(l, len);
                        int num = stoi(temp);
                        if (num >= 10 && num <= 26) {
                            f[i] = (f[i] + f[l]) % MOD;
                        }
                    }
                }
            }
        }

        return f[n];
    }
};

```

##### [LCR 165. 解密数字](https://leetcode.cn/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

> 思路：按照划分型DP的方法，f[i]表示前i个字符的解密方式个数，以最后一次的子字符串长度来划分。长度可以是1,2.假设最后一次的划分结果符合条件那么有`f[i]=f[i]+f[i-len]`  如果划分不成功，那么有`f[i]=f[i-1];`。

```c++
class Solution {
public:
    int crackNumber(int ciphertext) {
        string s=to_string(ciphertext);
        int n=s.size();
        vector<int> f(n+1,0);
        f[0]=1;
        for(int i=1;i<=n;i++){
            for(int len=1;len<=2;len++){
                int l=i-len,r=i;
                if(l<0)continue ;
                if(s[l]=='0')f[i]=f[i-1];
                else{
                    string temp=s.substr(l,len);
                    int num=stoi(temp);
                    if(num<0||num>25)continue;
                    f[i]+=f[l];
                }
                
            }
        }
        return f[n];
    }
};
```

##### [1416. 恢复数组](https://leetcode.cn/problems/restore-the-array/)

> 思路：使用划分型DP,定义f[i]表示为前i个字符的划分成功的方案数。属性为`sum`。由最后一次的划分来分割集合。集合可以划分为1,2,3,4…..k  其中`k<=i`。如果最后一次划分符合条件。那么有`f[i]+=f[i-k]`。
>
> ### 关键点
>
> 1. 对于前导0需要跳过。
> 2. 如果当前长度已经大于9，说明一定不能小于等于k，或者有前导0。直接`break`即可。
> 3. 如果当前最后一次的划分`num>k`说明不符合，直接`break`即可

```c++
class Solution {
public:
    static const int MOD = 1e9 + 7;

    int numberOfArrays(string s, int k) {
        int n = s.size();
        vector<int> f(n + 1, 0);  // 定义一个长度为n+1的数组f，其中f[i]表示前i个字符能被分割成的方案数
        f[0] = 1;  // 空字符串的分割方案数为1

        // 遍历每一个字符位置i
        for (int i = 1; i <= n; i++) {
            // 枚举最后一个数字的长度len
            for (int len = 1; len <= i; len++) {
                if (len > 10) break;  // 如果len大于10，则跳出循环，避免转换后的数字超过long long的范围

                int l = i - len;  // l为子串的起始位置
                if (s[l] == '0') continue;  // 跳过以'0'开头的数字

                string temp = s.substr(l, len);  // 提取长度为len的子串
                long long num = stoll(temp);  // 将子串转换为数字
                if (num > k) break;  // 如果数字大于k，跳出循环

                // 更新f[i]，将f[l]的值加到f[i]中，并取模
                f[i] = (f[i] + f[l]) % MOD;
            }
        }
        return f[n];  // 返回将整个字符串分割的方案数
    }
};

```

##### [2472. 不重叠回文子字符串的最大数目](https://leetcode.cn/problems/maximum-number-of-non-overlapping-palindrome-substrings/)

> 思路：依旧采用划分型DP，定义f[i]表示为前i个字符，能够划分成回文串的个数，属性为`Max`。由于需要判断每个区间是否为回文串，我们预处理一个`valid`数组，用来表示当前的区间[l,r]是否符合回文串的要求。如果不符合那么以第i个字符结尾的`f[i]`的值等于以第`i-1`个字符结尾的。集合可以分为0,k,k+1,……h其中h<=i。区间长度为0代表不分割，那么此时为`f[i]=f[i-1]`。其余为`f[i]=max(f[i],f[l]+1)`
>
> ### 关键点
>
> 1. 遇见回文串，有区间，一般先考虑预处理
> 2. 此题需要注意，区间的左端点为`i-len+1`,我们需要从上一个区间的右端点进行一个转移，即上一个区间的右端点为`l=i-len`

```c++
#include <vector>
#include <string>
#include <iostream>

using namespace std;

class Solution {
public:
    // 初始化一个快速判断区间回文的 DP 数组
    vector<vector<bool>> st;

    void palindrome(string &s) {
        int n = s.size();
        st.assign(n + 1, vector<bool>(n + 1, false));

        // 单个字符一定是回文
        for (int i = 1; i <= n; i++) 
            st[i][i] = true;

        // 两个字符的回文判断
        for (int i = 1; i < n; i++) {
            if (s[i - 1] == s[i])
                st[i][i + 1] = true;
        }

        // 枚举区间长度
        for (int len = 3; len <= n; len++) {
            for (int i = 1; i + len - 1 <= n; i++) {
                int l = i, r = i + len - 1;
                if (s[l - 1] == s[r - 1] && st[l + 1][r - 1]) {
                    st[l][r] = true;
                }
            }
        }
    }

    int maxPalindromes(string s, int k) {
        int n = s.size();
        st.assign(n + 1, vector<bool>(n + 1, false));
        palindrome(s);
        vector<int> f(n + 1, 0);

        for (int i = 1; i <= n; i++) {
            f[i] = f[i - 1]; // 继承 f[i-1] 的值
            for (int len = k; len <= i; len++) {
                int l = i - len + 1;//最后一个区间的左端点
                if (st[l][i]) {
                    f[i] = max(f[i], f[l - 1] + 1);//f[l-1]表示上一个区间的右端点进行转移
                }
            }
        }
        
        return f[n];
    }
};

```

##### [1105. 填充书架](https://leetcode.cn/problems/filling-bookcase-shelves/)

> 思路：依旧采用划分型DP，定义f[i]表示为前i本书放置后的高度，属性为`min`。由最后一次划分的子数组来分割集合。可以是1,2,3,4….k   k<=i，由于题目要求每一层的宽度不能超过`shelfWidth`。因此设置一个`sum`变量统计当前已经摆放的书的宽度。同时每一层的高度为高度最大的一本书，因此要设置一个`max_h`来记录当前最后一次划分的最大高度。状态转移方程为`f[i]=min(f[i],f[l]+max_h)`   l为上一次划分的右端点。

```c++
class Solution {
public:
    const int INF=0x3f3f3f3f;
    int minHeightShelves(vector<vector<int>>& books, int shelfWidth) {
        int n=books.size();
        vector<int> f(n+1,INF);
        f[0]=0;
        for(int i=1;i<=n;i++){
            int sum=0;//记录本次划分的宽度。
            int max_h=0;//记录本次划分的最大高度
            for(int len=1;len<=i;len++){
                int l=i-len;//上一次的右端点
                sum+=books[l][0];
                max_h=max(max_h,books[l][1]);
                if(sum>shelfWidth)break;
                f[i]=min(f[i],f[l]+max_h);
            }
        }
        return f[n];
    }
};
```

#### 约束划分的个数

##### [410. 分割数组的最大值](https://leetcode.cn/problems/split-array-largest-sum/)

> 思路：采用划分型DP，不过定义需要定义成二维,定义`f[i][j]`表示为前i个数字，分成j个子数组的和的最大值。属性为`min`。依旧采用最后一次子数组的长度来进行划分。可以分割为1,2,3,4….k。其中`k<=i`。由于需要计算的是子数组的最大值中最小。因此我们需要使用一个变量`sum`。进行记录最后一次划分的子数组和。得到了`sum`后。与`f[i-len][j-1]`对比来进行转移。可以得到转移方程`f[i][j]=min(max(f[i-len][j-1],sum))`  其中`1<=len<=i`
>
> ### 关键点
>
> 1. 初始化，`f[0][0]`表示为前0个数分成0个子数组和的最大值。因此`f[0][0]=0;`其他元素初始化为`INF`

```c++
class Solution {
public:
    const int INF=0x3f3f3f3f;
    int splitArray(vector<int>& nums, int k) {
        int n=nums.size();
        vector<vector<int>> f(n+1,vector<int>(k+1,INF));//定义f[i][j]表示前i个数字分成j个子数组的和的最大值。属性为min.

        //初始化
        f[0][0]=0;
        //枚举当前以第i个字符结尾
        for(int i=1;i<=n;i++){
            for(int j=1;j<=k;j++){
                //枚举长度
                int sum=0;
                for(int len=1;len<=i;len++){
                    int l=i-len,r=i;
                    sum+=nums[l];
                    f[i][j]=min(f[i][j],max(f[l][j-1],sum));
                }
            }
            
        }
        return f[n][k];
    }
};
```

##### [1043. 分隔数组以得到最大和](https://leetcode.cn/problems/partition-array-for-maximum-sum/)

> 思路：采用划分型DP，定义`f[i][j]`表示前i个数划分成j个子数组的最大值的和。属性为`Max`。使用最后一个子数组的长度来划分集合。可以划分为1,2,3…..k不能超过k，同时有效的长度应该要保证子数组的左端点应该是`>0`的因此可以提前剪枝。最后状态转移方程为`f[i][j]=max(f[i][j],f[i-len][j-1]+len*max_num)`。其中`max_num`为最后一个子数组的最大值。

```c++
class Solution {
public:
    int maxSumAfterPartitioning(vector<int>& arr, int k) {
        int n=arr.size();
        int max_num=*max_element(arr.begin(),arr.end());
        if(k==n)return max_num*n;
        vector<vector<int>> f(n+1,vector<int>(n+1,0));//定义f[i][j]表示前i个数字分成j个子数组的最大值和  属性为max
        //初始化
        f[0][0]=0;

        for(int i=1;i<=n;i++){
            //枚举子数组个数
            for(int j=1;j<=n;j++){
                //枚举最后一个子数组的长度
                int max_num=0;
                for(int len=1;len<=k;len++){
                    int l=i-len;
                    if(l<0)break;
                    max_num=max(max_num,arr[l]);
                    f[i][j]=max(f[i][j],f[i-len][j-1]+max_num*len);
                }
            }
        }
        return f[n][n];
    }
};
```

##### [1745. 分割回文串 IV](https://leetcode.cn/problems/palindrome-partitioning-iv/)

> 思路：采用划分型DP，定义f[i][j]表示为前i个字符能否分割成j个回文串。属性为`bool`。依旧采用最后一次的划分来分割集合。按照最后一次划分的长度，可以分为1,2,3,4…..k其中`k<=i`。最后一个区间的左端点为`L=i-len+1`。因此如果最后一次划分是回文串。那么有`f[i][j]=f[i][j]||f[l-1][j-1]`。

```c++
class Solution {
public:
    static const int N=2010;
    bool st[N][N];//判断区间i-j是否为回文串
    void palindrome(string &s){
        int n=s.size();
        memset(st,false,sizeof st);
        //初始化
        for(int i=1;i<=n;i++)st[i][i]=true;
        for(int i=1;i<n;i++)
            if(s[i]==s[i-1])st[i][i+1]=true;

        for(int len=3;len<=n;len++){
            //枚举区间左端点
            for(int i=1;i+len-1<=n;i++){
                int l=i,r=i+len-1;
                if(s[l-1]==s[r-1])
                st[l][r]=st[l][r]||st[l+1][r-1];
            }
        }
    }
    bool checkPartitioning(string s) {
        palindrome(s);
        int n=s.size();
        vector<vector<bool>> f(n+1,vector<bool>(4,false));//定义f[i][j]表示将前i个字符分割成j个回文子串能否成功，属性bool
        f[0][0]=true;//空串认为

        for(int i=1;i<=n;i++){
            //枚举分割个数
            for(int j=1;j<=3;j++){
                //枚举最后一次分割长度
                for(int len=1;len<=i;len++){
                    int l=i-len+1;
                    if(!st[l][i])continue;
                    f[i][j]=f[i][j]||f[l-1][j-1];//从倒数第二个区间右端点进行转移,l-1
                }
            }
        }
        return f[n][3];
    }
};
```

##### [813. 最大平均值和的分组](https://leetcode.cn/problems/largest-sum-of-averages/)

> 思路：使用划分型DP，定义f[i][j]表示为前i个数字划分成j个子数组的平均值和，属性为`max`。使用最后一次的子数组长度来划分集合。划分为1,2,3,4….len。`1<=len<=i`。同时对于状态转移方程有`f[i][j]=max(f[i-len+1][j-1]+sum/len)`
>
> ### 关键点
>
> 1. 由于此题需要计算区间的和，因此我们可以使用前缀和来进行优化。
> 2. 对于j==1的情况需要单独进行处理。

```c++
class Solution {
public:
    double largestSumOfAverages(vector<int>& nums, int k) {
        int n = nums.size();
        vector<vector<double>> f(n+1, vector<double>(k+1, 0)); // DP 数组，f[i][j] 表示前 i 个元素分成 j 个子数组的最大平均值和
        vector<double> s(n+1, 0); // 前缀和数组

        // 初始化前缀和数组
        for (int i = 1; i <= n; i++) {
            s[i] = s[i-1] + nums[i-1];
        }

        // 初始化当只有一个子数组时的情况
        for (int i = 1; i <= n; i++) {
            f[i][1] = s[i] / i;
        }

        // 动态规划过程
        for (int i = 1; i <= n; i++) { // 枚举当前处理到第 i 个元素
            for (int j = 2; j <= k; j++) { // 枚举分割的个数
                for (int len = 1; len <= i; len++) { // 枚举最后一个子数组的长度
                    int l = i - len + 1;
                    if (l > 0) { // 保证合法的子数组长度
                        f[i][j] = max(f[i][j], f[l-1][j-1] + (s[i] - s[l-1]) / len);
                    }
                }
            }
        }

        return f[n][k]; // 返回最终的最大平均值和
    }
};

```

##### [1278. 分割回文串 III](https://leetcode.cn/problems/palindrome-partitioning-iii/)

> 思路：采用划分型DP，定义`f[i][j]`表示为前i个字符划分成j个回文子串，需要修改的次数。属性为`min`，以最后一次划分的子串长度来分割集合。集合可以划分为1,2,3,4….len，其中`1<=len<=i`。最后一个区间的左端点表示为`L=i-len+1`。状态表示为`f[i][j]=min{f[L-1][j-1]+update_count}`。`update_count`表示为区间`[L,i]`变为回文串所需要修改的最小次数。此题我们可以初始化一个DP数组，`DP[i][j]`表示为区间i-j变为回文串所需要修改的最小次数。则状态转移方程变为`f[i][j]=min{f[L-1][j-1]+dp[L][i]}`
>
> ### 关键点
>
> 1. 预处理一个DP数组，表示`DP[i][j]`表示为区间i-j变为回文串所需要修改的最小次数。

```c++
class Solution {
public:
    static const int N=110;
    int dp[N][N];//dp[i][j]表示将i-j子串变为回文串的修改次数，属性为min
   
    void palindrome(string &s){
        int n=s.size();
        memset(dp,0x3f,sizeof dp);
        //初始化
        dp[0][0]=0;
        //单个字符不需要修改
        for(int i=1;i<=n;i++)
            dp[i][i]=0;
        
        //两个字符
        for(int i=1;i<n;i++){
            if(s[i-1]!=s[i])
            dp[i][i+1]=1;
            else
            dp[i][i+1]=0;
        }
        for(int len=3;len<=n;len++){
            //枚举左端点
            for(int i=1;i+len-1<=n;i++){
                int l=i,r=i+len-1;
                if(s[l-1]==s[r-1])
                dp[l][r]=dp[l+1][r-1];
                else
                dp[l][r]=dp[l+1][r-1]+1;
            }
        }
    }
    int palindromePartition(string s, int k) {
        palindrome(s);
        int n=s.size();
        vector<vector<int>> f(n+1,vector<int>(k+1,0x3f));
        //初始化
        f[0][0]=0;

        for(int i=1;i<=n;i++){
            for(int j=1;j<=k;j++){
                //枚举最后一次划分长度
                for(int len=1;len<=i;len++){
                    int l=i-len+1;
                    f[i][j]=min(f[i][j],f[l-1][j-1]+dp[l][i]);
                }
            }
        }
        return f[n][k];

    }
};
```

##### [1335. 工作计划的最低难度](https://leetcode.cn/problems/minimum-difficulty-of-a-job-schedule/)

> 思路：使用划分型DP，定义`f[i][j]`表示为将前i个数字分成j个子数组的最大难度之和。属性为`min`。使用最后一次的划分的长度来分割集合。集合可以划分为1,2,3,4,5…len  其中`1<=len<=i`。使用一个`max_num`来记录最后一次划分数组的最大元素。最后一次划分数组的左端点为`L=i-len+1`。状态转移方程为`f[i][j]=min{f[l-1][j-1]+max_num}`
>
> ### 关键点
>
> 1. 由于求的属性为最小值`min`。因此初始化`f[0][0]=0`,其余为`INF`

```c++
class Solution {
public:
     const int INF=0x3f3f3f3f;
    int minDifficulty(vector<int>& nums, int d) {
        int n=nums.size();
        if (n < d) return -1; // 如果天数大于元素个数，无法划分，直接返回 -1
        vector<vector<int>> f(n+1,vector<int>(d+1,INF));
        //初始化
        f[0][0]=0;
        for(int i=1;i<=n;i++){
            for(int j=1;j<=d;j++){
                int max_num=0;
                for(int len=1;len<=i;len++){
                    int l=i-len+1;
                    max_num=max(max_num,nums[l-1]);
                    f[i][j]=min(f[i][j],f[l-1][j-1]+max_num);
                }
            }
        }
        return f[n][d]>=INF/2?-1:f[n][d];
    }
};
```





### 数位DP问题

#### [2719. 统计整数数目](https://leetcode.cn/problems/count-of-integers/)

> 思路：利用模板，此题的前导0对答案无影响，不用使用`is_num`。需要额外的两个全局变量`max_sum`,`min_sum`枚举到了数字长度后判断当前是否满足条件`sum`在区间`[min_sum,max_sum]`。记忆化时，以枚举了前`i`个数，同时前`i`个数的和为`sum`作为条件。定义`memo[i][sum]`表示满足条件的数有多少个。

```c++
class Solution {
public:
    int len;
    string num;
    int max_;
    int min_;
    vector<vector<int>> memo;
    static const int MOD=1e9+7;
    int dfs(int i,int sum,bool is_limit){
        if(sum>max_)return 0;
        if(i==len)
            return sum>=min_&&sum<=max_;
        if(!is_limit&&memo[i][sum]!=-1)return memo[i][sum];
        int res=0;
        int up=is_limit?(num[i]-'0'):9;
        
        for(int j=0;j<=up;j++){
            res=(res+dfs(i+1,sum+j,is_limit&&(j==up)))%MOD;
        }
        if(!is_limit)memo[i][sum]=res;
        return res;

    }
    int count(string s){
        num=s;
        len=num.size();
        memo=vector<vector<int>> (len+1,vector<int>(500,-1));
        return dfs(0,0,true);
    }
    int count(string num1, string num2, int min_sum, int max_sum) {
        max_=max_sum;
        min_=min_sum;
        int res1=count(num2);
        int res2=count(num1);
        int res=(res1-res2+MOD)%MOD;
        int sum=0;
        for(char c:num1)
            sum+=c-'0';
        return res+(sum>=min_sum&&sum<=max_sum);
    }
};
```

#### [788. 旋转数字](https://leetcode.cn/problems/rotated-digits/)

> 思路：查看题意，发现满足要求的数必须包含`2,5,6,9`其中一个，不能包含`3,4,7`。因此我们定义`memo[i][has]`表示前i个数中是否包含`2,5,6,9`条件下，满足条件的数。最后合格的数必须要满足`is_num&&has`。此题的前导0对答案有影响，例如0020跟20不等价。因此要使用`is_num`。选择数字时候如果遇见了`3,4,7`直接跳过即可。

```c++
class Solution {
public:
    string num;
    int len;
    vector<vector<int>> memo;
    int dfs(int i,bool has,bool is_limit,bool is_num){
        if(i==len)
            return is_num&&has;
        if(!is_limit&&is_num&&memo[i][has]!=-1)return memo[i][has];
        int res=0;
        if(!is_num)
            res+=dfs(i+1,false,false,false);
        int up=is_limit?(num[i]-'0'):9;
        
        for(int j=1-is_num;j<=up;j++){
            if(j==2||j==5||j==6||j==9)
                res+=dfs(i+1,true,is_limit&&j==up,true);
            else if(j==0||j==1||j==8)
                res+=dfs(i+1,has,is_limit&&j==up,true);
            else
                continue;
        }
        if(!is_limit&&is_num)memo[i][has]=res;
        return res; 
    }
    int count(int n){
        num=to_string(n);
        len=num.size();
        memo=vector<vector<int>> (len+1,vector<int>(3,-1));
        return dfs(0,false,true,false);
    }
    int rotatedDigits(int n) {
        return count(n);
    }
};
```

#### [902. 最大为 N 的数字组合](https://leetcode.cn/problems/numbers-at-most-n-given-digit-set/)

> 思路：本题前导0没有影响，但是依旧不能填0，因为`digits`数组中不含0，如果构造出了类似`102`这样中间有0的情况，是不符合要求的。因此需要使用`is_num`。最后的需要满足是挑选了数字，因此满足条件为`is_num!=false`。定义`memo[i]`表示，**在不受到 \*n\* 的约束时**的合法方案数

```c++
class Solution {
public:
    string num;
    int len,max_;
    vector<int> memo;
    int dfs(int i,bool is_limit,bool is_num,vector<string> &digits){
        if(i==len)
            return is_num!=false;
        if(!is_limit&&is_num&&memo[i]!=-1)return memo[i];
        int res=0;
        if(!is_num)
            res+=dfs(i+1,false,false,digits);
        int up=is_limit?(num[i]-'0'):9;
        for(string c:digits){
            int d=(c[0]-'0');
            if(d>up)break;
            res+=dfs(i+1,is_limit&&d==up,true,digits);
        }
        if(!is_limit&&is_num)memo[i]=res;
        return res;
    }
    int count(int n,vector<string> &digits){
        num=to_string(n);
        len=num.size();
        memo=vector<int> (len+1,-1);
        return dfs(0,true,false,digits);
    }
    int atMostNGivenDigitSet(vector<string>& digits, int n) {
        max_=n;
        return count(n,digits);
    }
};
```

#### [233. 数字 1 的个数](https://leetcode.cn/problems/number-of-digit-one/)

> 思路：本题的前导0对于答案无影响，我们需要记录构造整数中1的个数，因此，需要额外的参数`cnt`表示前i个数中1的个数。
>
> 定义`memo[i][cnt]`表示为满足条件前i个数中，1的个数为`cnt`的方案数。

```c++
class Solution {
public:
    string num;
    int len;
    vector<vector<int>> memo;
    int dfs(int i,int cnt,bool is_limit){
        if(i==len)
            return cnt;
        if(!is_limit&&memo[i][cnt]!=-1)return memo[i][cnt];
        int res=0;
        int up=is_limit?(num[i]-'0'):9;
        for(int j=0;j<=up;j++){
            res+=dfs(i+1,cnt+(j==1),is_limit&&j==up);
        }
        if(!is_limit)memo[i][cnt]=res;
        return res;
    }
    int count(int n){
        num=to_string(n);
        len=num.size();
        memo=vector<vector<int>> (len+1,vector<int> (10,-1));
        return dfs(0,0,true);
    }
    int countDigitOne(int n) {
        return count(n);
    }
};
```

#### [600. 不含连续1的非负整数](https://leetcode.cn/problems/non-negative-integers-without-consecutive-ones/)

> 思路：此题的前导0对答案无影响，由于前后两个数位有关系，因此额外定义参数`last`,表示上一个数位是什么。定义`memo[i][last]`表示满足条件前i个数，最后一个数位为last的个数。

```c++
class Solution {
public:
    vector<int> num;
    int len;
    vector<vector<int>> memo;
    int dfs(int i,int last,bool is_limit){
        if(i==len)
            return 1;
        if(!is_limit&&memo[i][last]!=-1)return memo[i][last];
        int res=0;
        int up=is_limit?(num[i]):1;
        for(int j=0;j<=up;j++){
            if(j==last&&j==1)continue;
            res+=dfs(i+1,j,is_limit&&j==up);
        }
        if(!is_limit)memo[i][last]=res;
        return res;
    }
    int count(int n){
        while(n){
            num.emplace_back(n%2);
            n/=2;
        }
        len=num.size();
        reverse(num.begin(),num.end());
        memo=vector<vector<int>> (len+1,vector<int> (2,-1));
        return dfs(0,0,true);
    }
    int findIntegers(int n) {
        return count(n);
    }
};
```

#### [2376. 统计特殊整数](https://leetcode.cn/problems/count-special-integers/)

> 思路：此题的前导0对于答案有影响，例如0010,与10相同。根据题意，条件为每个数位均不相同，因此我们可以使用状态压缩，定义mask表示每一个数位选择的数的集合。定义`memo[i][mask]`表示为满足条件前i个数，选择状态为`mask`的个数。
>
> ### 重点
>
> 1. 是否在集合  `(mask>>j)&1`
> 2. 加入集合     `mask|(1<<j)`

```c++
class Solution {
public:
    string num;
    int len;
    vector<vector<int>> memo;
    int dfs(int i,int mask,bool is_limit,bool is_num){
        if(i==len)
            return is_num;
        if(!is_limit&&memo[i][mask]!=-1)return memo[i][mask];
        int res=0;
        if(!is_num) 
            res+=dfs(i+1,0,false,false);
        int up=is_limit?(num[i]-'0'):9;
        for(int j=1-is_num;j<=up;j++){
            if(((mask>>j)&1)!=0)continue;
            res+=dfs(i+1,mask|(1<<j),is_limit&&j==up,true);
        }
        if(!is_limit&&is_num)memo[i][mask]=res;
        return res;
    }
    int count(int n){
        num=to_string(n);
        len=num.size();
        int mask=1<<10;
        memo=vector<vector<int>> (len+10,vector<int>(mask+8,-1));
        return dfs(0,0,true,false);
    }
    int countSpecialNumbers(int n) {
        return count(n);
    }
};
```

#### [2827. 范围中美丽整数的数目](https://leetcode.cn/problems/number-of-beautiful-integers-in-the-range/)

> 思路：本题前导0对答案有影响，要使用`is_num`。同时根据条件，我们需要额外的定义`sum`,以及`even,odd`。定义`memo[i][sum][even][odd]`表示为满足条件，前i个数，和为sum能被k整除，奇数数目为even，偶数数目为odd的个数。由于本题求的是区间，因此转换为`f[r]-f[l-1]`即可

```c++
class Solution {
public:
    vector<vector<vector<vector<int>>>> memo;
    int len;
    string num;
    int mod;

    int dfs(int i,  bool is_limit, bool is_num, int ou, int ji, long long sum) {
        if (i == len)
            return ou == ji && is_num && sum % mod == 0;  // 判断是否为美丽整数

        if (!is_limit && is_num && memo[i][sum][ou][ji] != -1)  // 使用相对偏移处理 ou 和 ji
            return memo[i][sum][ou][ji];

        int res = 0;

        if (!is_num)  // 可以跳过当前位
            res += dfs(i + 1, false, false, ou, ji, sum);

        int up = is_limit ? (num[i] - '0') : 9;  // 限制当前位的上限

        for (int j = 1 - is_num; j <= up; ++j) {
            int new_ou = ou + !(j % 2);  // 更新偶数位的数量
            int new_ji = ji + (j % 2);   // 更新奇数位的数量
            res += dfs(i + 1, is_limit && (j == up), true, new_ou, new_ji, (sum * 10 + j)%mod);
        }

        if (!is_limit && is_num) 
            memo[i][sum][ou][ji] = res;

        return res;
    }

    int countBeautifulNumbers(int n) {
        num = to_string(n);
        len = num.size();
        memo = vector<vector<vector<vector<int>>>>(len, vector<vector<vector<int>>>(mod+5, vector<vector<int>>(11, vector<int>(11,-1))));  // 使用偏移处理差值
        return dfs(0, true, false, 0, 0, 0);
    }

    int numberOfBeautifulIntegers(int low, int high, int k) {
        mod = k;

        // 计算 [1, high] 的美丽整数数量
        int res_high = countBeautifulNumbers(high);

        // 计算 [1, low-1] 的美丽整数数量
        int res_low = countBeautifulNumbers(low - 1);

        // 返回两者之差，即为 [low, high] 范围内的美丽整数数量
        return res_high-res_low;
    }
};

```

#### [2801. 统计范围内的步进数字数目](https://leetcode.cn/problems/count-stepping-numbers-in-range/)

> 思路：本题的前导0对答案有影响，0010与10不同，需要使用`is_num`。本题关系为相邻两个之间的关系，因此额外定义一个参数`last`。`memo[i][last]`表示，满足条件前i个数，且最后一个数为`last`的个数。同时在枚举数字的时候，如果遇到了前面已经选了数字，并且当前选择数字与前面步长不为1，则跳过即可。

```c++
class Solution {
public:
    string num;
    int len;
    vector<vector<int>> memo;
    static const int MOD=1e9+7;
    int dfs(int i,int last,bool is_limit,bool is_num){
        if(i==len)
            return is_num;
        if(!is_limit&&is_num&&memo[i][last]!=-1)return memo[i][last];
        int res=0;
        if(!is_num)
            res+=dfs(i+1,0,false,false);
        int up=is_limit?(num[i]-'0'):9;
        for(int j=1-is_num;j<=up;j++){
            int step=abs(j-last);
            if(is_num&&step!=1)continue;
            res=(res+dfs(i+1,j,is_limit&&(j==up),true))%MOD;
        }
        if(!is_limit&&is_num)memo[i][last]=res;
        return res;
    }
    int count(string n){
        num=n;
        len=num.size();
        memo=vector<vector<int>> (len+1,vector<int> (11,-1));
        return dfs(0,0,true,false);
    }
    bool judge(string s){
        int n=s.size();
        for(int i=0;i<n-1;i++)
            if(abs(s[i]-s[i+1])!=1)return false;
        return true;
    }
    int countSteppingNumbers(string low, string high) {
        int res1=count(high);
        int res2=count(low);
        return (res1-res2+MOD+judge(low))%MOD;
    }
};
```

### 状态机DP

#### [121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

> 思路：使用状态机DP，发现每一个元素拥有两种状态，即当前是否持有股票。则可以利用这两个状态进行一个转移。定义f[i][j]表示第i天结束后，当前持有股票的状态为j的情况下，手上拥有的金额。可以得到状态转移方程为,`f[i][0]=max(f[i-1][0],f[i-1][1]+w[i-1])  f[i][1]=max(f[i-1][1],-w[i-1]);`解释：如果第i天结束没有股票，那么有两种情况，第i天不买股票，或者第i天将股票卖出。如果第i天有股票，那么即第i天买股票，或者是在前面买的。注：本题**仅能出手一次**彩票
>
> 最后由于我们卖出的情况下可以取得最大值，因此返回`f[n][0]`即为最大值。
>
> ### 关键点
>
> 1. `f[0][1]=-INF;`此时的状态为非法，不能够到达，第0天此时还没有股票，不可能拥有。

```c++
class Solution {
public:
    static const int INF=0x3f3f3f3f;
    int maxProfit(vector<int>& w) {
        int n=w.size();
        vector<vector<int>> f(n+1,vector<int>(2,0));
        f[0][0]=0;
        f[0][1]=-INF;
        for(int i=1;i<=n;i++){
            f[i][0]=max(f[i-1][0],f[i-1][1]+w[i-1]);
            f[i][1]=max(f[i-1][1],-w[i-1]);
        }
        return f[n][0];
    }
};
```

#### [122. 买卖股票的最佳时机 II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/)

> 思路：题目条件为最多拥有一只股票，但是可以不停的买卖，因此定义状态f[i][j]表示为第i天结束后，状态为j的情况下手里拥有的金钱。分析与上题一致。不过对于手里有票的转移与上题不一致，此题是从前一天转移过来的。因此`f[i][1]=max(f[i-1][1],f[i-1][0]-w[i])`。同样最后手里没有股票时金额最大。
>
> ### 关键点
>
> 1. `f[0][1]=-INF;`此时的状态为非法，不能够到达，第0天此时还没有股票，不可能拥有。

```c++
class Solution {
public:
    static const int INF=0x3f3f3f3f;
    int maxProfit(vector<int>& w) {
        int n=w.size();
        vector<vector<int>> f(n+1,vector<int>(2,0));
        f[0][0]=0;
        f[0][1]=-INF;//此时的状态为非法，不能够到达，第0天此时还没有股票，不可能拥有。
        for(int i=1;i<=n;i++){
            f[i][0]=max(f[i-1][0],f[i-1][1]+w[i-1]);
            f[i][1]=max(f[i-1][1],f[i-1][0]-w[i-1]);
        }
        return f[n][0];
    }
};
```

#### [123. 买卖股票的最佳时机 III](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iii/)

> 思路：此题要求最多完成两笔交易，因此我们要细分为三个状态，未完成交易，完成一次交易，完成`>=2`次交易。状态定义为 `f[i][j][k]` 表示为第i天结束后完成了j次交易是否持有股票手里拥有的金额。分为以下情况。
>
> 0表示完成了0次交易，1表示完成了1次交易，2表示完成了2次交易。
>
> 状态转移有0-0,0-1,1-1,1-2,2-2
>
> ### 关键点
>
> 1. 由于执行的交易次数有了限制，因此我们需要额外定义一个维度来表示当前进行了几次交易。
> 2. 初始化：由于`f[i][0][0]`此时未进行交易，同时未持有股票。赋值为0，其余元素赋值为-INF



```c++
class Solution {
public:
    // 定义常量 INF 和数组大小 N
    static const int INF = 0x3f3f3f3f, N = 1e5 + 10;
    
    // 状态转移数组 f[i][j][k]
    // f[i][j][0] 表示第 i 天结束时，完成了 j 次交易且不持有股票的最大金额
    // f[i][j][1] 表示第 i 天结束时，完成了 j 次交易且持有股票的最大金额
    int f[N][3][2];
    
    int maxProfit(vector<int>& w) {
        int n = w.size();
        
        // 初始化状态数组 f
        memset(f, -0x3f, sizeof f);  // 将所有值初始化为负无穷大
        
        // 初始化0次交易的状态
        // 在第0天，未完成任何交易且不持有股票，最大金额为0
        for (int i = 0; i <= n; i++) f[i][0][0] = 0;
        
        // 状态转移
        for (int i = 1; i <= n; i++) {  // 遍历每天
            for (int j = 1; j <= 2; j++) {  // 遍历交易次数
                // 更新第i天完成j次交易且不持有股票的最大金额
                // 有两种情况：不买股票，或卖出股票
                f[i][j][0] = max(f[i - 1][j][0], f[i - 1][j][1] + w[i - 1]);
                
                // 更新第i天完成j次交易且持有股票的最大金额
                // 有两种情况：不买股票，或买入股票
                f[i][j][1] = max(f[i - 1][j][1], f[i - 1][j - 1][0] - w[i - 1]);
            }
        }
        
        // 计算结果
        int res = 0;
        // 只考虑在完成最多两笔交易且不持有股票的情况下的最大金额
        for (int i = 0; i <= 2; i++)
            res = max(res, f[n][i][0]);
        
        return res;
    }
};

```

#### [188. 买卖股票的最佳时机 IV](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iv/)

> 思路：此题对于交易次数进行了进一步的限制，最多进行k次交易。同样需要额外一维来记录进行的交易次数。定义`f[i][j][k]`表示为第`i`天结束后完成了`j`次交易是否持有股票手里拥有的金额。此题的最大值为`max{f[n][j][0]}`其中$0<=j<=k$​
>
> ### 关键点
>
> 1. 除了0次交易，其他的状态都可以由自己转移或者上一个状态转移。因此初始化0次交易的状态。

```c++
class Solution {
public:
    static const int N=1010,K=110,INF=0x3f3f3f3f;
    int f[N][K][2];//定义为到了第i天，完成了j次交易，手上状态为k
    int maxProfit(int k, vector<int>& w) {
        int n=w.size();
        memset(f,-INF,sizeof f);
        for(int i=0;i<=n;i++)f[i][0][0]=0;

        for(int i=1;i<=n;i++){
            for(int j=1;j<=k;j++){
                f[i][j][0]=max(f[i-1][j][0],f[i-1][j][1]+w[i-1]);
                f[i][j][1]=max(f[i-1][j][1],f[i-1][j-1][0]-w[i-1]);
            }
        }
        int res=0;
        for(int i=0;i<=k;i++)res=max(res,f[n][i][0]);
        return res;
    }
};
```

#### [309. 买卖股票的最佳时机含冷冻期](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

> 思路：此题有一个冷冻期，翻译过来即为，卖了股票第二天不能买，因此需要隔一天，我们需要分为三个状态即，手中持有股票（定义状态为0），手中无股票第1天（定义状态为1），手中无股票`>=`2天（定义状态为2）。转移的过程可以为0-0,0-1,1-2,2-2,2-0
>
> 转移方程为按照状态机边来转移即可。
>
> ### 关键点
>
> 1. 初始化入口：  ` f[0][1]=0` `f[0][2]=0`除了第0天没有持有股票的时候金额为0，其他都是非法状态即为-INF
> 2. 出口，出口可以为第1天无股票或者>=2天无股票，因此为`max(f[n][1],f[n][2])`

```c++
class Solution {
public: 
    static const int N=5050,INF=0x3f3f3f3f;
    int f[N][3];//定义f[i][j]表示为交易到了第i天，手中股票的状态，0表示有股票，1表示第一天没有股票，2表示超过1天没有股票
    int maxProfit(vector<int>& w) {
        int n=w.size();
        memset(f,-INF,sizeof f);

        //初始化，入口为
        f[0][1]=0;
        f[0][2]=0;

        for(int i=1;i<=n;i++){
            f[i][0]=max(f[i-1][0],f[i-1][2]-w[i-1]);
            f[i][1]=f[i-1][0]+w[i-1];
            f[i][2]=max(f[i-1][2],f[i-1][1]);
        }
        return max(f[n][1],f[n][2]);
    }
};
```

#### [714. 买卖股票的最佳时机含手续费](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

> 思路：本题需要在出售股票的时候添加手续费，其余过程与买卖股票2一致，分为两种状态，手中有股票，手中无股票。状态转移为 0-0,0-1,1-1,1-0.
>
> ### 关键点：
>
> 1. 需要在卖出股票时支付手续费。
> 2. 入口为`f[0][0]=0`表示为当前第0天是没有股票的同事也没有持有股票，金额为0，其余状态为非法状态，设为`-INF`

```c++
class Solution {
public:
    static const int N=5e4+10,INF=0x3f3f3f3f;
    int f[N][2];
    int maxProfit(vector<int>& w, int x) {
        int n=w.size();
        memset(f,-INF,sizeof f);
        //初始化
        f[0][0]=0;
        f[0][1]=-INF;

        for(int i=1;i<=n;i++){
            f[i][0]=max(f[i-1][0],f[i-1][1]+w[i-1]-x);
            f[i][1]=max(f[i-1][1],f[i-1][0]-w[i-1]);
        }
        return f[n][0];
    }
};
```

#### [1493. 删掉一个元素以后全为 1 的最长子数组](https://leetcode.cn/problems/longest-subarray-of-1s-after-deleting-one-element/)

> 思路1：不定长滑动窗口，保持窗口内部仅有一个0
>
> 思路2：状态机DP，定义`f[i][j]`表示当前删除了j个元素的情况下以nums[i]结尾的全为1的最长子数组。分类讨论。
>
> 1. 当前元素为0
>
>    1. 未删除元素，那么有`f[i][0]=f[i-1][0];`
>    2. 删除了元素，此时长度为0，即`f[i][1]=0`
>
> 2. 当前元素为1
>
>    1. 未删除情况，有`f[i][0]=f[i-1][0]+1`
>    2. 删除情况，有`f[i][1]=f[i-1][1]=1；`
>
>    返回值取`max{f[i][1]} `   $0<=i<=n$

```c++
class Solution {
public:
    static const int N=1e5+10,INF=0x3f3f3f3f;
    int f[N][2];
    int longestSubarray(vector<int>& nums) {
        int n=nums.size();
        memset(f,-INF,sizeof f);
        //初始化
        f[0][0]=0;
        f[0][1]=-1;
        int res=0;
        for(int i=1;i<=n;i++){
           if(nums[i-1]==0){
                f[i][0]=0;
                f[i][1]=f[i-1][0];
           }else{
                f[i][0]=f[i-1][0]+1;
                f[i][1]=f[i-1][1]+1;
           }
           res=max(res,f[i][1]);
        }
        return res;
    }
};
```

#### [2745. 构造最长的新字符串](https://leetcode.cn/problems/construct-the-longest-new-string/)

> 思路：题目要求不能出现`AAA`,`BBB`。定义`f[i][j][k][l]`表示使用了`i`个`AA``,j`个`BB`,`k`个`AB`的情况下结尾为状态`l`的字符串的最大长度。分情况讨论
>
> 1. 结尾为`AA`
>    1. 后面可以跟BB
> 2. 结尾为`BB`
>    1. 后面可以跟AB,AA
> 3. 结尾为`AB`
>    1. 后面可以跟AB,AA
>
> 返回值为整个`f[i][j][k][l]`的最大值。

```c++
class Solution {
public:
    static const int N=52;
    int f[N][N][N][4];
    int longestString(int x, int y, int z) {
        // memset(f,-0x3f,sizeof f);
        // f[0][0][0][0]=0;
        // f[0][0][0][1]=0;
        // f[0][0][0][2]=0;
        //初始化
        int res=0;
        for(int i=0;i<=x;i++){
            for(int j=0;j<=y;j++){
                for(int k=0;k<=z;k++){
                    if(i>0)
                    f[i][j][k][0]=max(f[i-1][j][k][1],f[i-1][j][k][2])+2;
                    if(j>0)
                    f[i][j][k][1]=f[i][j-1][k][0]+2;
                    if(k>0)
                    f[i][j][k][2]=max(f[i][j][k-1][1],f[i][j][k-1][2])+2;
                    for(int s=0;s<3;s++)
                        res=max(res,f[i][j][k][s]);
                }
            }
        }
        return res;
    }
};
```

#### [2222. 选择建筑的方案数](https://leetcode.cn/problems/number-of-ways-to-select-buildings/)

> 思路：题目要求选择3栋建筑，同时相邻的建筑不能相同，得出结论，相邻元素之间有关系，因此，我们需要去记录上一次选择为什么类型的建筑。同时，分类讨论
>
> 1. 当前建筑为0
>    1. 从前i个中选择，j个，结尾为0的方案数：`f[i][j][0]=f[i-1][j][0]+f[i-1][j-1][1]`。分别是（放弃`s[i]`，直接从左边取`k`栋结尾为`0`)，以及（选取`s[i]`，左边选`k-1`栋且结尾为`1`不造成冲突）
>    2. 从前i个中选择，j个，结尾为1的方案数：`f[i][j][1]=f[i-1][j][1]`表示为（放弃`s[i]`，直接从左边选`k`栋结尾为`1`）
> 2. 当前建筑为1
>    1. `f[i][j][0]=f[i-1][j][0]`与上面分析类似
>    2. `f[i][j][1]=f[i-1][j][1]+f[i-1][j-1][0]`
>
> 最后答案为`f[n][3][0]+f[n][3][1]`
>
> ### 关键点
>
> 1. 初始化入口，`f[0][0][0]=1,f[0][0][1]=1`表示为不选房子的方案数只有一种。即空序列。
> 2. 单独处理选择0个房子的情况，即`f[i][0][1]=f[i-1][0][1]`。和`f[i][0][0]=f[i-1][0][0]`

```c++
#include <vector>
#include <string>
using namespace std;

class Solution {
public:
    static const int N = 1e5 + 10;
    long long f[N][4][2] = {};

    long long numberOfWays(string s) {
        int n = s.size();
        // 初始化：不选字符时，只有一种方法，即空序列
        f[0][0][0] = 1;
        f[0][0][1] = 1;

        for (int i = 1; i <= n; i++) {
            f[i][0][1]=f[i-1][0][1];
            f[i][0][0]=f[i-1][0][0];
            for (int j = 1; j <= 3; j++) {
                // 当前字符为 '0'
                if (s[i-1] == '0') {
                    f[i][j][0] = f[i-1][j][0] + f[i-1][j-1][1];
                    f[i][j][1] = f[i-1][j][1];
                }
                // 当前字符为 '1'
                else {
                    f[i][j][1] = f[i-1][j][1] + f[i-1][j-1][0];
                    f[i][j][0] = f[i-1][j][0];
                }
            }
        }

        return f[n][3][0] + f[n][3][1];
    }
};
```

#### [376. 摆动序列](https://leetcode.cn/problems/wiggle-subsequence/)

> 思路：采用LCS的思路来思考，定义`f[i][j]`表示以`nums[i]`结尾，状态为j的子序列，共有两种状态，当前为上升状态，当前为下降状态。分类讨论如下。考虑子序列倒数第2的点的位置，可以为`1,2,3,4….i-1`。
>
> 1. 当前nums[i]>nums[j]
>    1. `f[i][1]`:`f[i][1]=max(f[i][1],f[j][0]+1)`（倒数第二次的状态必须为下降的状态。）
>    2. `f[i][0]`：`f[i][0]=max(f[i][0],f[j][0])`（等同于倒数第二次结尾的长度）
> 2. nums[i]<nums[j]
>    1. `f[i][1]`: `f[i][1]=max(f[i][1],f[j][1])`（由于跟到属于第二次状态一致，因此长度不能增加，只能和倒数第二次j相同。）
>    2. `f[i][0]：f[i][0]=max(f[i][0],f[j][1]+1)`（本次状态为下降的状态，要求倒数第二次状态为上升）
>
> 返回结果为`max(f[n][0],f[n][1])`
>
> ### 关键点
>
> 1. 初始化入口`,f[0][0],f[0][1]=0`,空字符长度为0，`f[1][0] = 1;  f[1][1] = 1;`单个字符也是摆动的序列

```c++
class Solution {
public:
    static const int N=1e3+10; // 定义数组大小上限
    int f[N][2]; // 动态规划数组，f[i][0]表示以nums[i-1]结尾的下降摆动序列的长度，f[i][1]表示以nums[i-1]结尾的上升摆动序列的长度

    int wiggleMaxLength(vector<int>& nums) {
        int n = nums.size(); // 获取输入数组的大小
        
        // 初始化
        f[0][0] = 0; // f[0][0]和f[0][1]初始化为0，表示起始状态
        f[0][1] = 0;
        f[1][0] = 1; // f[1][0]和f[1][1]初始化为1，表示单个元素也是一个摆动序列
        f[1][1] = 1;

        int res = 1; // 记录最长摆动序列的长度，初始为1（单个元素）

        for(int i = 2; i <= n; i++) { // 从第二个元素开始遍历
            for(int j = 1; j < i; j++) { // 枚举每个元素i之前的所有元素
                if(nums[i-1] > nums[j-1]) { // 如果nums[i-1]大于nums[j-1]
                    f[i][1] = max(f[i][1], f[j][0] + 1); // 更新以nums[i-1]结尾的上升摆动序列的长度
                    f[i][0] = max(f[j][0], f[i][0]); // 保持下降序列的最大值不变
                } else if(nums[i-1] < nums[j-1]) { // 如果nums[i-1]小于nums[j-1]
                    f[i][1] = max(f[i][1], f[j][1]); // 保持上升序列的最大值不变
                    f[i][0] = max(f[i][0], f[j][1] + 1); // 更新以nums[i-1]结尾的下降摆动序列的长度
                }
            }
            res = max({res, f[i][0], f[i][1]}); // 更新全局最长摆动序列的长度
        }

        return res; // 返回最长摆动序列的长度
    }
};







```

#### [1567. 乘积为正数的最长子数组长度](https://leetcode.cn/problems/maximum-length-of-subarray-with-positive-product/)

> 思路：定义f[i][j]表示为以nums[i]结尾的子数组状态为j的最值乘积长度。分为最大和最小。分类讨论
>
> 1. nums[i]>0
>    1. `f[i][0]`：`f[i][0]=f[i-1][0]>0?(f[i-1][0]+1):0`表示为，如果前一个数`nums[i-1]`结尾的负数乘积长度不为0，那么即可加1，否则为0
>    2. `f[i][1]：f[i][1]=f[i-1][1]+1`。当前为整数，长度为以nums[i-1]结尾的子数组长度+1
> 2. nums[i]<0
>    1. `f[i][0]：f[i-1][1]+1`表示为前面以nums[i-1]结尾的整数乘积+1即可
>    2. `f[i][1]：f[i-1][0]>0?(f[i-1][0]+1):0`表示为当前为负数，如果前面以nums[i-1]结尾的子数组负数乘积长度不为0，那么即可+1，否则为0；
> 3. nums[i]==0
>    1. 二者都需置为0`f[i][0]=f[i][1]=0;`
>
> 返回值为max{f[i][1]},$1<=i<=n$
>
> ### 关键点
>
> 1. 初始化入口需要将第一个数的子数组长度判断，即有如果为负数，`f[i][0]=1`如果为整数，`f[i][1]=1;`

```c++
class Solution {
public:
    static const int N=1e5+10; // 定义数组大小的上限
    int f[N][3]; // 动态规划数组，f[i][0]表示以第i个元素结尾的负数乘积子数组的长度，f[i][1]表示以第i个元素结尾的正数乘积子数组的长度

    int getMaxLen(vector<int>& nums) {
        int n = nums.size(); // 获取输入数组的大小
        
        // 初始化
        f[1][0] = nums[0] < 0 ? 1 : 0; // 如果第一个元素为负数，f[1][0]初始化为1，否则为0
        f[1][1] = nums[0] > 0 ? 1 : 0; // 如果第一个元素为正数，f[1][1]初始化为1，否则为0
        
        int res = f[1][1]; // 记录当前乘积为正数的最长子数组的长度
        
        for(int i = 2; i <= n; i++) { // 从第二个元素开始遍历
            if(nums[i-1] > 0) { // 当前元素为正数的情况
                f[i][0] = f[i-1][0] > 0 ? (f[i-1][0] + 1) : 0; // 如果前一个元素结尾的负数子数组长度大于0，则更新为f[i-1][0] + 1，否则为0
                f[i][1] = f[i-1][1] + 1; // 正数子数组的长度加1
            } else if(nums[i-1] < 0) { // 当前元素为负数的情况
                f[i][0] = f[i-1][1] + 1; // 负数子数组的长度加1
                f[i][1] = f[i-1][0] > 0 ? (f[i-1][0] + 1) : 0; // 如果前一个元素结尾的负数子数组长度大于0，则更新为f[i-1][0] + 1，否则为0
            } else { // 当前元素为0的情况
                f[i][0] = f[i][1] = 0; // 重置，因为乘积为0
            }
            res = max(res, f[i][1]); // 更新乘积为正数的最长子数组长度
        }
        
        return res; // 返回最终的结果
    }
};

```



#### [2708. 一个小组的最大实力值](https://leetcode.cn/problems/maximum-strength-of-a-group/)

> 思路：本题的要求是求子序列的最大乘积，定义f[i][j]表示为以nums[i]结尾的状态为j的最值乘积，最大和最小。分类讨论。
>
> 1. 当小组不包括当前元素，小组的最大实力值和最小实力值分别为`f[i-1][1]`和` f[i-1][0]`。
> 2. 当小组包含元素 `nums[i]` 时，小组的最大实力值和最小实力值分别为 `f[i−1][1]×nums[i]、f[i−1][0]×nums[i]` 和 `nums[i]` 三项中的最大值和最小值。
> 3. 状态转移方程为
> 4. `f[i][1]`=max(`f[i-1][1]`,`f[i−1][1]`×nums[i],`f[i−1][0]`×nums[i],nums[i])，
> 5. `f[i−1][0]`=min(`f[i−1][0]`,`f[i−1][1]`×nums[i],`f[i−1][0]`×nums[i],nums[i])。

```c++
class Solution {
public:
    long long maxStrength(vector<int>& nums) {
        int n=nums.size();
        vector<vector<long long>> f(n+1,vector<long long>(2,0));

        f[1][0]=nums[0];//最小值
        f[1][1]=nums[0];//最大值
        for(int i=2;i<=n;i++){
            long long x=nums[i-1];
            f[i][0]=min({x,f[i-1][0]*x,f[i-1][1]*x,f[i-1][0]});
            f[i][1]=max({x,f[i-1][1]*x,f[i-1][0]*x,f[i-1][1]});
        }
        return f[n][1];
    }
};
```

#### [2826. 将三个组排序](https://leetcode.cn/problems/sorting-three-groups/)

> 思路1：采用反向思考，计算整个数组的最长非递减子序列长度。时间复杂度为O(n^2)
>
> 思路2：采用状态机DP，题目翻译过来为，修改数组中的元素使整个数组变为非递减的最小次数。定义`f[i][j]`表示为将第i个数变为`j`的总次数。前面均满足条件。可以进行修改的值为,`[j,3]`。$1=<j<=3$。如果当前的nums[i]跟当前要变成的值相同，那么就不用进行修改了，修改次数为0.





#### [2786. 访问数组中的位置使分数最大](https://leetcode.cn/problems/visit-array-positions-to-maximize-score/)

> 思路：按照题目要求，定义`f[i][j]`表示为以nums[i]即为的子序列，值为状态j的最大值。分类讨论
>
> 1. 当前为偶数
>    1. 选`f[i][0]=max(f[i-1][0],f[i-1][1]-x)+nums[i]`（相同奇偶性的，与不同奇偶性-x）取最大值
>    2. 不选`f[i][1]=f[i-1][1]`（最大值跟上一次奇偶性相同的一致）
> 2. 当前为奇数
>    1. 选`f[i][1]=max(f[i-1][1],f[i-1][0]-x)+nums[i]`（相同奇偶性，与不同奇偶性-x）取最大值
>    2. 不选`f[i][0]=f[i-1][0]`（最大值跟上一次奇偶性相同的一致）
>
> 返回值为`max(f[n][0],f[n][1])`
>
> ### 关键点
>
> 1. 初始化入口，如果第一个数为奇数，则`f[1][1]=nums[0]`,如果为偶数,则有`f[1][1]=nums[0]`。其余情况为非法，设置为-INF

```c++
class Solution {
public:
    long long maxScore(vector<int>& nums, int x) {
        int n=nums.size();
        vector<vector<long long>> f(n+1,vector<long long>(2,-0x3f3f3f3f));
        if(nums[0]%2==0){
            f[1][0]=nums[0];
        }else
            f[1][1]=nums[0];
        for(int i=2;i<=n;i++){
            if(nums[i-1]%2==0){
                //偶数情况
                f[i][0]=max(f[i-1][0],f[i-1][1]-x)+nums[i-1];
                f[i][1]=f[i-1][1];
            }else{
                //奇数情况
                f[i][1]=max(f[i-1][1],f[i-1][0]-x)+nums[i-1];
                f[i][0]=f[i-1][0];
            }
        }
        return max(f[n][0],f[n][1]);
    }
};
```

#### [1262. 可被三整除的最大和](https://leetcode.cn/problems/greatest-sum-divisible-by-three/)

> 思路：被三整除余数为0,1,2，总共会有三个状态。定义f[i][j]表示以nums[i]结尾的子序列，状态为j的最大和。由于每个数的取余3的余数固定有范围。分类讨论。`f[i][j]`表示为以`nums[i]`结尾的序列余数为j的最大和。
>
> 1. 如果余数为0，
>    1. `f[i][0]=f[i][0]+nums[i]`
>    2. `f[i][1]=f[i-1][1]+nums[i];`
>    3. `f[i][2]=f[i-1][2]+nums[i]`
> 2. 余数为1
>    1. `f[i][0]=max(f[i-1][0],f[i-1][2]+num);`
>    2. `f[i][1]=max(f[i-1][1],f[i-1][0]+num);`
>    3. `f[i][2]=max(f[i-1][2],f[i-1][1]+num);`         
>
> 3. 余数位2
>
>    1. `f[i][0]=max(f[i-1][0],f[i-1][1]+num);`
>
>    2. `f[i][1]=max(f[i-1][1],f[i-1][2]+num);`
>
>    3. `f[i][2]=max(f[i-1][2],f[i-1][0]+num);`
>
> 4. 合并为`f[i][j]=max(f[i-1][j],f[i-1][(j+mod)%3]+x);`
>
> ### 关键点
>
> 1. 初始化入口为`f[0][0]=0`表示为不选数余数为0的最大和为0，其余为非法=-INF

```c++
class Solution {
public:
    static const int N=4e4+10;
    int f[N][3];
    int maxSumDivThree(vector<int>& nums) {
        int n=nums.size();
        f[0][1]=0;
        f[0][1] = INT_MIN;  // 负无穷
        f[0][2] = INT_MIN;  // 负无穷
        for(int i=1;i<=n;i++){
            int x=nums[i-1];
            int mod=nums[i-1]%3;
            for(int j=0;j<=2;j++){
                f[i][j]=max(f[i-1][j],f[i-1][(j+mod)%3]+x);
            }
        }
        return f[n][0];
    }
};
```

#### [1911. 最大子序列交替和](https://leetcode.cn/problems/maximum-alternating-subsequence-sum/)

> 思路：定义`f[i][j]`表示`f[i][0]`表示前 i个数中以偶数下标结尾的子序列的最大交替和，`f[i][1]`表示前 i 个数中以奇数下标结尾的子序列的最大交替和。分类讨论。对于第i个数有选和不选两种策略
>
> 1. 当前元素在子序列中下标为偶数
>
>    `f[i][0]=max(f[i-1][0],f[i-1][1]-nums[i])`（表示为不选，和选）
>
> 2. 当前元素在子序列中下标为奇数
>
>    `f[i][1]=max(f[i-1][1],f[i-1][0]+nums[i])`（表示为不选和选）
>
> 最后答案为`max(f[n][0],f[n][1])`
>
> 
>
> ### 关键点
>
> 1. 初始化入口，`f[0][0]=0,f[0][1]=-INF`其余为-INF
> 2. 注：上述我们让下标从1开始了，因此对于第一个数的下标为1其实对应为下标为0，也就是偶数。因此在结尾为偶数时候需要减去当前数。

```c++
class Solution {
public:
    static const int N=1e5+10;
    long long f[N][2];
    long long maxAlternatingSum(vector<int>& nums) {
        int n=nums.size();
        // 初始化
        memset(f,0,sizeof f);
        f[0][0]=0;
        f[0][1]=-0x3f3f3f3f;//表示前0个数，下标为奇数的情况，不存在这种情况，因此为非法状态
        for(int i=1;i<=n;i++){
            f[i][1]=max(f[i-1][1],f[i-1][0]+nums[i-1]);
            f[i][0]=max(f[i-1][0],f[i-1][1]-nums[i-1]);
        }
        return max(f[n][1],f[n][0]);
    }
};

```

#### [1395. 统计作战单位数](https://leetcode.cn/problems/count-number-of-teams/)

> 思路：题目翻译过来，求长度为3的子序列，递增或者递减。的方案数。定义f[i][j][k]表示f[i][j][0]表示为前i个数，选择j个，序列为递减的方案数，f[i][j][1]表示为前i个数，选择j个，序列为单调递增的方案数。分类讨论。由倒数第二次选择的位置来进行划分，对于当前位置数字，有选和不选两种方案。
>
> 1. 当前为递增，nums[i]>nums[j]
>    1. `f[i][j]][1]+=f[i-1][j-1][1]`
> 2. 当前为递减 nums[i]<nums[j]
>    1. `f[i][j][0]+=f[i-1][j-1][0]`
>
> 最后返回sum{`f[i][3][0]+f[i][3][1]`}
>
> ### 关键点
>
> 1. 每个单个数字都是长度递增或者递减的子序列。

```c++
class Solution {
public:
    static const int N=1010;
    int f[N][4][2];
    int numTeams(vector<int>& nums) {
        int n=nums.size();
        //初始化
        for(int i=1;i<=n;i++){
            f[i][1][0]=1;
            f[i][1][1]=1;
            for(int j=1;j<i;j++){
                //上升//下降
                
                for(int k=2;k<=3;k++){
                    if(nums[i-1]>nums[j-1])
                    f[i][k][0]+=f[j][k-1][0];
                    else if(nums[i-1]<nums[j-1])
                    f[i][k][1]+=f[j][k-1][1];
                }  
            }
        }
        int res=0;
        for(int i=1;i<=n;i++)
            res+=f[i][3][0]+f[i][3][1];
        return res;
    }
};
```

#### [2771. 构造最长非递减子数组](https://leetcode.cn/problems/longest-non-decreasing-subarray-from-two-arrays/)

> 思路：此题要求由两个数组同一个位置元素，每次挑选一个，问能够构成的最长的非递减序列。定义`f[i][j]`表示为`f[i][0]`表示以nums1[i]结尾的最长长度，`f[i][1]`表示为以nums2[i]结尾的最长长度。分类讨论，讨论当前选择哪个元素，讨论上一个选择的元素属于哪个数组。
>
> 1. 选择nums1
>    1. 上一个为nums2     满足条件，则有，`f[i][0]=f[i-1][1]+1`
>    2. 上一个为nums1      满足条件，则由  `f[i][0]=f[i-1][0]+1  `             二者取最大值
> 2. 选择nums2
>    1. 上一个为nums2     满足条件，则有，`f[i][1]=f[i-1][1]+1`
>    2. 上一个为nums1      满足条件，则由  `f[i][1]=f[i-1][0]+1`         二者取最大值
>
> 返回值为，所有f中的最大值
>
> ### 关键点
>
> 1. 本题与最大子数组和为一个类型。都是跟相邻的元素有关系
> 2. 初始化，单个数字长度为1

```c++
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int maxNonDecreasingLength(vector<int>& nums1, vector<int>& nums2) {
        int n = nums1.size();
        vector<vector<int>> f(n+1, vector<int>(2, 1));//单个数字的长度为1

        int res = 1;
        for (int i = 2; i <=n; i++) {
            //当前选择的为nums1
            if(nums1[i-1]>=nums1[i-2])
            f[i][0]=f[i-1][0]+1;
            if(nums1[i-1]>=nums2[i-2])
            f[i][0]=max(f[i][0],f[i-1][1]+1);//二者的最大值
            //当前选择为nums2
            if(nums2[i-1]>=nums1[i-2])
            f[i][1]=f[i-1][0]+1;
            if(nums2[i-1]>=nums2[i-2])
            f[i][1]=max(f[i][1],f[i-1][1]+1);
            res=max({res,f[i][0],f[i][1]});
        }
        return res;
    }
};
```

#### [1186. 删除一次得到子数组最大和](https://leetcode.cn/problems/maximum-subarray-sum-with-one-deletion/)

> 思路：考虑当前元素能否被删除，定义`f[i][j]`表示为`f[i][0]`表示，以`nums[i]`结尾的子数组，删除个数为0的最大和，f[i][1]表示为，以`nums[i]`结尾的子数组，删除个数为1的最大和。讨论当前元素nums[i]能否被删除,以之前是否删除过元素来进行讨论
>
> 1. 之前删除过元素
>
>    1. 当前元素只能够选不删除   `f[i][1]=f[i-1][1]+x `         
>
> 2. 之前没有删除过元素
>
>    1. 当前元素被删除  ` f[i][1]=f[i-1][0]`
>    2. 当前元素不删除  `f[i][0]=f[i-1][0]+x`
>    3. 以nums[i]为左端点   `f[i][0]=x`
>
> 3. 综上有，
>
>    1. `f[i][0]=max(f[i-1][0],0)+nums[i-1];`
>    2. `f[i][1]=max(f[i-1][0],f[i-1][1]+nums[i-1]);`
>
>    ​       返回值为f中的最大值
>
> ### 关键点
>
> 1. 初始化 ` f[0][0]=0,f[0][1]=-INF`
> 2. 遇见这种要删除一个数的子数组，我们通过去枚举前面是否删除过数字，以及跟最大子数组和联系起来，考虑当前元素，是否能够接在上一个元素的后面。
> 3. 由于子数组必须有一个元素，因此比较大小时要跳过`f[1][1]`

```c++
class Solution {
public:
    static const int N = 1e5+10, INF = 0x3f3f3f3f;  // 定义常量 N 为数组最大长度，INF 为一个非常大的正数，用于初始化最小值
    int f[N][2];  // 定义动态规划数组 f，f[i][0] 表示以第 i 个元素结尾且未删除元素的最大子数组和，f[i][1] 表示以第 i 个元素结尾且已删除一个元素的最大子数组和

    int maximumSum(vector<int>& nums) {
        int n = nums.size();  // 获取数组长度
        int maxSum = -INF;  // 初始化 maxSum 为一个非常小的值，用于记录最大子数组和
        f[0][0] = 0;  // f[0][0] 表示以第 0 个元素结尾且未删除元素的最大子数组和，因为没有元素，所以为 0
        f[0][1] = -INF;  // f[0][1] 表示以第 0 个元素结尾且已删除一个元素的最大子数组和，不可能一开始就删除元素，所以初始化为负无穷

        for(int i = 1; i <= n; i++) {  // 从第 1 个元素开始遍历
            f[i][0] = max(f[i-1][0], 0) + nums[i-1];  // 计算 f[i][0]：要么延续之前的子数组（f[i-1][0]），要么重新开始一个新子数组（从 0 开始），然后加上当前元素 nums[i-1]
            f[i][1] = max(f[i-1][0], f[i-1][1] + nums[i-1]);  // 计算 f[i][1]：要么在之前未删除元素的子数组中删除当前元素（f[i-1][0]），要么继续在之前已删除一个元素的子数组中加入当前元素（f[i-1][1] + nums[i-1]）
            maxSum = max({maxSum, f[i][0], i > 1 ? f[i][1] : -INF});  // 更新 maxSum，考虑 f[i][0] 和 f[i][1]，其中 f[i][1] 只有在 i > 1 时才有效
        }

        return maxSum;  // 返回最大子数组和
    }
};

```

#### [1594. 矩阵的最大非负积](https://leetcode.cn/problems/maximum-non-negative-product-in-a-matrix/)

> 思路1：暴搜
>
> 思路2：利用状态机DP，有与乘积有正有负，我们需要定义`f[i][j][k]`表示为`f[i][j][0]`表示从(0,0)点到`(i,j)`点的最小乘积,`f[i][1]`表示为从`(0,0)`点到`(i,j)`点的最大乘积。枚举状态转移即可，总共只能从上方和左边转移，因此枚举4个乘积即可。
>
> ### 关键点
>
> 1. 由于第一行第一列特殊，因此要对其特殊处理初始化

```c++
class Solution {
public:
    static const int M = 16, N = 16;
    const int MOD = 1e9 + 7;
    long long f[M][N][2]; // f[i][j][0] - max product, f[i][j][1] - min product

    int maxProductPath(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();

        // 初始化第一行和第一列
        f[1][1][0] = f[1][1][1] = grid[0][0];
        for (int i = 2; i <= n; i++) {
            f[1][i][0] = f[1][i][1] = f[1][i-1][0] * grid[0][i-1];
        }
        for (int i = 2; i <= m; i++) {
            f[i][1][0] = f[i][1][1] = f[i-1][1][0] * grid[i-1][0];
        }

        // 动态规划填表
        for (int i = 2; i <= m; i++) {
            for (int j = 2; j <= n; j++) {
                int x = grid[i-1][j-1];
                f[i][j][0] = max({f[i-1][j][0]*x, f[i][j-1][0]*x,f[i-1][j][1]*x,f[i][j-1][1]*x}) ;
                f[i][j][1] = min({f[i-1][j][1] * x, f[i][j-1][1]* x,f[i-1][j][0]*x, f[i][j-1][0]*x});
            }
        }

        // 结果
        long long result = f[m][n][0];
        return result < 0 ? -1 : result%MOD;
    }
};

```

### 树形DP问题

#### 树的直径

> [!IMPORTANT]
>
> 树的直径概念：在一棵树中，两节点之间的**最长**路径的长度



##### [543. 二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/)

> 思路：本题是一个树形DP的简单应用，对于求二叉树的直径，我们考虑以每个点作为分叉点的最大长度。定义dfs(i)表示为以当前根节点i为分叉点的最大长度，也即为max(left+1,right+1)。我们求的是边权，也就是边，每个子树距离父节点的边权值为1，因此我们要+1,。每个节点都有可能为分叉点，因此我们要进行全局更新答案。最后返回当前节点作为分叉点的最大长度。

```c++
class Solution {
public:
    int res=0;
    int dfs(TreeNode *root){
        if(root==nullptr)
            return -1;
        int lh=dfs(root->left)+1;
        int rh=dfs(root->right)+1;
        int len=lh+rh;
        res=max(res,len);
        return max(lh,rh);
    }
    int diameterOfBinaryTree(TreeNode* root) {
        dfs(root);
        return res;
    }
};
```

##### [124. 二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum/)

> 思路：本题求得是最大的路径和，也是一道树的直径题目，依旧以每个点作为分叉点来尝试更新最大的路径和。如果当前点作为分叉点的路径长度为负数，那么返回0即可，意味着不经过该点一定更优。此题我们求得是点权型，因此遇到null返回0即可。
>
> 1. 关键点
>    1. dfs(i)表示为从底部一个叶子节点出发，以当前顶点作为分叉点向下的最大路径和。
>    2. 返回值，如果以当前点作为分叉点的路径和为负数，那么代表不选择当前顶点一定更优，返回0即可。`max(max(l_sum,r_sum)+root->val,0)`

```c++
class Solution {
public:
    int res=-0x3f3f3f3f;
    int dfs(TreeNode *root){
        if(root==nullptr)
            return 0;
        int sum_l=dfs(root->left);
        int sum_r=dfs(root->right);

        res=max(res,sum_l+sum_r+root->val);
        return max(max(sum_l,sum_r)+root->val,0);//如果当前节点作为分叉点的最长路径和为负数，那么返回0即可。
    }
    int maxPathSum(TreeNode* root) {
        dfs(root);
        return res;
    }
};
```

##### [2246. 相邻字符不同的最长路径](https://leetcode.cn/problems/longest-path-with-different-adjacent-characters/)

> 思路：本题与树的直径一致，不过本题有了额外的限制，要求相邻路径点不能相同。同时本题为一个树，因此我们需要去保存每个节点的孩子节点。与图的建图一致。
>
> 1. 关键点
>    1. 我们不能将判断相同的条件写在最前面，因此每个节点都是有可能作为分叉点的，如果写在了最前面，那么会跳过一条分支的节点。会造成错误的答案。
>    2. 此题求得是点的个数，因此我们需要再求边的个数的基础上+1即可。

```c++
class Solution {
public:
    int res=0;
    int  dfs(int x,vector<vector<int>> &p,string &s){
        int x_len=0;
        for(int y:p[x]){
            int y_len=dfs(y,p,s)+1;
            if(s[y]==s[x])continue;//不能写在y_len的前面。
            res=max(res,x_len+y_len);
            x_len=max(x_len,y_len);
        }
        return x_len;
    }
    int longestPath(vector<int>& parent, string s) {
        int n=parent.size();
        vector<vector<int>> p(n+1);
        //初始化邻接表
        for(int i=1;i<n;i++){
            p[parent[i]].emplace_back(i);
        }
        dfs(0,p,s);
        return res+1;
    }   
};
```

##### [687. 最长同值路径](https://leetcode.cn/problems/longest-univalue-path/)

> 思路：本题求的是相同值的路径长度，因此，我们必须要保证相邻的两个节点之间的值相同，如果相同，那么左子树/右子树的链长即为本身，否则为不符合条件，为0.
>
> 1. 关键点
>    1. 回顾dfs(i)代表的含义，表示当前节点为分界点的最长的路径长度。
>    2. 如果相邻不符合条件，说明左子树或者右子树的链长为0；

```c++
class Solution {
public:
    int res=0;
    int dfs(TreeNode *root){
        if(root==nullptr)
            return -1;
        int l_h=dfs(root->left)+1;
        int r_h=dfs(root->right)+1;
        
        //不符合条件
        if(root->left&&root->left->val!=root->val)
            l_h=0;
        if(root->right&&root->right->val!=root->val)
            r_h=0;
        res=max(res,l_h+r_h);
        return max(l_h,r_h);
    }
    int longestUnivaluePath(TreeNode* root) {
    dfs(root);
    return res;
    }
};
```

##### [3203. 合并两棵树后的最小直径](https://leetcode.cn/problems/find-minimum-diameter-after-merging-two-trees/)

>  思路：树的直径为树中任意两点之间路径长度的最大值，两颗树合并后的直径，最小直径有三种情况
>
> 1. 第一颗树很大，第二颗树很小，那么合并后的直径为第一颗树的直径。
>
> 2. 第二棵树很大，第一棵树很小，合并后的直径为第二棵树的直径
>
> 3. 选择两棵树直径的中点添加过后，那么新树的最小直径为右边直径的一半加上左边直径一半再加上新链接点的长度为1
>
>    即$(d1+1)/2+(d2+1)/2+1$​
>
>    最后返回三者的最大值即为新树的最小直径。
>
> * 关键点
>   * 题目所给为无向图，因此我们建树时候需要添加两条边。
>   * 在计算当前顶点为根是，需要判断邻接点是否为父亲。避免发生错误。

```c++
class Solution {
public:
    int res1=0;
    int dfs(int x,int fa,vector<vector<int>> &p){
        int x_len=0;
        for(int y:p[x]){
            if(y==fa)continue;
            int y_len=dfs(y,x,p)+1;
            res1=max(res1,x_len+y_len);
            x_len=max(x_len,y_len);
        }
        return x_len;
    }
    int  mask(vector<vector<int>> &e){
        res1=0;
        int n=e.size();
        vector<vector<int>> p(n+1);
        //建树
        for(int i=0;i<n;i++){
            p[e[i][0]].emplace_back(e[i][1]);
            p[e[i][1]].emplace_back(e[i][0]);
        }
           dfs(0,-1,p);
        return res1;
    }
    int minimumDiameterAfterMerge(vector<vector<int>>& edges1, vector<vector<int>>& edges2) {
            int d1=mask(edges1);
            int d2=mask(edges2);
            return max({d1,d2,(d1+1)/2+(d2+1)/2+1});
    }
};
```

##### [1617. 统计子树中城市之间最大距离](https://leetcode.cn/problems/count-subtrees-with-max-distance-between-cities/)

> 思路：本题为树的直径和子集的结合。我们需要去枚举每一个子树，计算每个子树的直径。需要保证子树是连通的。
>
> 对于每个子树，我们需要做的有两个
>
> 1. 检测树的连通性：使用一个`vis`数组，`vis[i]`表示在计算直径的过程中，是否已被访问过。
> 2. 计算树的直径。：利用dfs(i)表示为当前点为分叉点的最大往下链长。
>
> ### 关键点
>
> 1. 枚举子集，我们枚举的子集表示为当前选中节点构成的子树，使用`in_set`来表示。
> 2. 如果检测树的连通性?如果最后`dfs(i)`计算子树的直径完毕，那么`vis`中的节点选中情况，一定跟`in_set`中的情况一致。因此我们仅需比较`vis`是否等于`in_set`。即`vis==in_set`
> 3. 枚举子集完成后，我们仅从当前选中的子树中的一个顶点出发，就可以知道整颗子树的直径。不需要遍历所有在`in_set`中的节点。会造成重复访问。因此遍历一个就可以。
> 4. 优化，n很小，我们可以使用状态压缩，利用一个s来代表子集。

```c++
//解法1，利用子集
class Solution {
public:
    vector<int> res; // 存储每个直径对应的子树数量
    vector<int> vis; // 记录节点是否被访问过
    vector<int> in_set; // 记录当前枚举的子集
    int dimeter_max = 0; // 记录当前子集的最大直径
    vector<vector<int>> g; // 邻接矩阵，用于存储树的结构

    // 深度优先搜索，用于计算子树的直径
    int dfs(int x) {
        vis[x] = true; // 标记当前节点已被访问过
        int x_len = 0; // 记录从节点 x 出发的最长路径长度
        for (int y : g[x]) { // 遍历 x 的所有邻接点 y
            // 如果 y 已经访问过，或者不在当前子集中，跳过
            if (vis[y] || !in_set[y]) continue;
            // 计算从 y 出发的最长路径
            int y_len = dfs(y) + 1;
            // 更新当前子集的最大直径
            dimeter_max = max(dimeter_max, x_len + y_len);
            // 更新从 x 出发的最长路径
            x_len = max(x_len, y_len);
        }
        return x_len;
    }

    // 递归枚举所有可能的子集
    void ziji(int u, int n) {
        if (u == n) {
            // 枚举完成，开始计算当前子集的直径
            for (int v = 0; v < n - 1; v++) {
                if (!in_set[v]) continue; // 跳过不在子集中的节点
                fill(vis.begin(), vis.end(), 0); // 清空访问记录
                dimeter_max = 0; // 重置最大直径
                dfs(v); // 从节点 v 开始计算直径
                break; // 只需从一个在子集中的节点开始计算即可
            }
            // 如果子集是连通的，且最大直径不为 0，则记录结果
            if (dimeter_max && vis == in_set)
                res[dimeter_max - 1]++;
            return;
        }

        // 不选节点 u，继续递归下一个节点
        ziji(u + 1, n);

        // 选择节点 u，加入子集，继续递归下一个节点
        in_set[u] = true;
        ziji(u + 1, n);
        in_set[u] = false; // 还原现场，移除节点 u
    }

    vector<int> countSubgraphsForEachDiameter(int n, vector<vector<int>>& edges) {
        g.resize(n); // 初始化邻接矩阵
        for (int i = 0; i < n - 1; i++) {
            int a = edges[i][0] - 1, b = edges[i][1] - 1; // 节点编号从 1 转为 0 开始
            g[a].emplace_back(b); // 建立无向图
            g[b].emplace_back(a);
        }
        res.resize(n - 1); // 初始化结果数组，存储不同直径的子树数量
        vis.resize(n); // 初始化访问标记数组
        in_set.resize(n); // 初始化子集标记数组
        ziji(0, n); // 开始枚举所有子集
        return res; // 返回结果
    }
};


//状态压缩，子集优化
class Solution {
public:
    vector<int> res;           // 保存每个直径的子树数量
    int dimeter_max = 0;       // 当前子树的最大直径
    vector<vector<int>> g;     // 邻接表表示的图

    // 计算整数 x 的二进制表示中从最低位到最高位有多少个 '1'，返回第一个 '1' 的位置（从0开始）
    int count_one(int x) {
        int count = 0;
        while ((x & 1) == 0) {  // 找到 x 的最低位 '1'
            x >>= 1;
            count++;
        }
        return count;
    }

    // 深度优先搜索，计算从节点 x 出发的子树的最大直径
    int dfs(int x, int &vis, int mask) {
        vis |= (1 << x);  // 标记当前节点 x 为已访问
        int x_len = 0;
        for (int y : g[x]) {  // 遍历 x 的所有邻接点 y
            // 如果 y 已访问过，或者 y 不在当前子集 mask 中，跳过
            if (((vis >> y) & 1) != 0 || ((mask >> y) & 1) == 0) continue;
            int y_len = dfs(y, vis, mask) + 1;  // 递归计算 y 的子树长度
            dimeter_max = max(dimeter_max, x_len + y_len);  // 更新当前子树的最大直径
            x_len = max(x_len, y_len);  // 更新从 x 出发的最长路径
        }
        return x_len;
    }

    vector<int> countSubgraphsForEachDiameter(int n, vector<vector<int>>& edges) {
        g.resize(n);
        // 初始化邻接表
        for (int i = 0; i < n - 1; i++) {
            int a = edges[i][0] - 1, b = edges[i][1] - 1;
            g[a].emplace_back(b);
            g[b].emplace_back(a);
        }
        res.resize(n - 1);  // 结果数组，保存直径为 1 到 n-1 的子树数量

        // 枚举所有可能的子集 mask，3 表示至少包含两个节点的子集
        for (int mask = 3; mask < (1 << n); mask++) {
            if ((mask & (mask - 1)) == 0) continue; // 需要至少两个点
            int vis = 0;
            dimeter_max = 0;  // 重置当前子集的最大直径
            int start_node = count_one(mask);  // 从最低位的 '1' 开始 DFS
            dfs(start_node, vis, mask);  // 计算当前子集的直径
            if (vis == mask)  // 如果所有节点都被访问过，说明是连通子集
                res[dimeter_max - 1]++;  // 增加该直径对应的子树计数
        }
        return res;
    }
};

```

##### [2538. 最大价值和与最小价值和的差值](https://leetcode.cn/problems/difference-between-maximum-and-minimum-price-sum/)

> 思路：本题是树形DP比较经典的题目，对于此题，我们翻译过来即为，删除的点一定是一条路径的端点（度为1），因此，答案总共有两种情况，第一种是，当前节点为分支节点，其他的子树分支，不删除节点的最大值加上当前分支删除底部的端点的和，或者为其他子树分支，删除一个节点的最大值加上当前分支不删除底部端点的和。对于二者取`max`即可。
>
> 我们可以返回两个值，一个为删除叶子节点分支的最大值，一个为不删除叶子节点分支的最大值。那么既有。其他分支删除和不删除叶子节点的两种情况，和当前分支删除和不删除两种情况。最大值为`max(其他不删+当前删，其他删+当前不删)`。
>
> 同时需要维护当前x为根的最大删分支，跟最大不删分支。也就是，**其他删/不删，当前删/不删比较。**

```c++
class Solution {
public:
    typedef pair<long long, long long> PLL;
    vector<vector<int>> g;
    long long res = 0;
    vector<int> w; 

    PLL dfs(int x, int fa) {
        long long max_no = w[x]; // 未删
        long long max_yes = 0;   // 已删
        for (int y : g[x]) {
            if (y == fa) continue;
            auto [s_no, s_yes] = dfs(y, x);
            // 更新结果: 其他删 + 当前不删，其他不删 + 当前删
            res = max({res, max_no + s_yes, max_yes + s_no});
            // 维护删除和不删除的最大分支,当前分支跟其他分支做比较。
            max_no = max(max_no, s_no + w[x]);
            max_yes = max(max_yes, s_yes + w[x]);//如果能够进到这个循环，说明不是叶子节点，要加上。
        }
        return {max_no, max_yes};
    }

    long long maxOutput(int n, vector<vector<int>>& edges, vector<int>& price) {
        g.resize(n); // 节点编号从 0 到 n-1
        for (auto e : edges) {
            int a = e[0], b = e[1];
            g[a].emplace_back(b);
            g[b].emplace_back(a);
        }

        this->w = price;
        dfs(0, -1);
        return res;
    }
};

```

##### [2385. 感染二叉树需要的总时间](https://leetcode.cn/problems/amount-of-time-for-binary-tree-to-be-infected/)

> 思路：本题依旧是树形DP应用，我们题目要求出最大的感染时间，其实也就是求的以star为分叉点的最大分支链长。考虑两个问题，将整个树进行一个拆分，分为以star为根的树，以及star为叶子节点的树。
>
> * 第一个问题：以star为根的树，最长的感染时间取决于它的子树的最大链长。
> * 第二个问题：star为叶子节点的树。此时求得是整个子树包含start的直径。（即从 `start` 节点到树中其他节点的最大路径长度）
>
> ### 做法
>
> 1. 定义`dfs(root)`返回两个值，第一个值为当前节点作为分叉点的最大链长，第二个值返回是否包含了start这个点。
> 2. 分为4种情况
>    1. 节点为`null`，此时返回`{-1,false}`表示不包含`start`点。返回-1是为了保证叶子节点的链长为`0`；
>    2. 节点为`start`，更新答案为左右两个分支的最大链长。即`res=max(lh,rh);`
>    3. 左右子树都不包含`star`，返回当前节点分叉的最大链长即可。即`max(lh,rh);`
>    4. 左右子树包含链长，此时表示第二种情况，需要更新答案，表示当前节点作为分叉点的最大感染时间，**为不包含start的最大链长加上包含start作为端点的链长**。即`res=max(res,lh+rh)`。我们需要返回包含子树的链长。即`{flag_l?lh:rh,true}`

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
typedef pair<int,bool> PIB;
class Solution {
public:
    int res=0,start;
    PIB dfs(TreeNode *root){
        if(root==nullptr)
            return {-1,false};
        auto [lh,flag_l]=dfs(root->left);
        auto [rh,flag_r]=dfs(root->right);
        lh++;
        rh++;
        if(root->val==start){//如果当前为开始点,更新答案为最长的一条链.
            res=max(lh,rh);
            return {0,true};
        }
        if(flag_l||flag_r){//如果其中一条路径包含start，也就是start作为端点,返回包含start的链长。
            res=max(res,lh+rh);//更新答案为，当前节点作为分叉点的不包含start的链长，加上包含start的链长。总共两个链，因此直接相加即可。
            return {flag_l?lh:rh,true};
        }
        
        
        return {max(lh,rh),false};
    }
    int amountOfTime(TreeNode* root, int start) {
        this->start=start;
        dfs(root);
        return res;
    }
};
```
