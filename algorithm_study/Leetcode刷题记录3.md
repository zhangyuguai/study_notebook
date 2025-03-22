## 14.贪心

### 区间贪心

#### 不相交区间

##### [646. 最长数对链](https://leetcode.cn/problems/maximum-length-of-pair-chain/)

> 思路1：由于此题的数据量较少，我们可以使用`LIS`来做，由于我们寻找的为最长数对链的个数，定义`f[i]`表示为以第i个区间结尾的长度。属性为`Max`。我们以倒数第二个区间的右端点来进行划分集合，可以划分为 `0,1,2,3….j`。`（0<=j<i）`如果当前第i个区间的左端点大于第j个区间的右端点，那么有 `f[i]=max(f[i],f[j]+1)`，初始化f数组默认值为1，表示单个区间也符合。最后的答案为，`max(f)`
>
> 思路2：采用贪心的做法，我们将区间按照左端点进行一个从小到大的排序，要想取得最长的长度，那么每次需要保证区间右端点足够小，因此我们每次选择最小的区间右端点，如果当前区间的左端点大于上一个区间右端点，说明没有重合，长度加1，否则的话，说明重合，长度不变，尝试选择二者较小的一个区间右端点。保证选择的一定是当前最小的区间右端点。

```c++
class Solution {
public:
    struct range {
        int l, r;
        bool operator<(const range &R) const {
            return l < R.l;  // 按照左端点排序
        }
    };

    int findLongestChain(vector<vector<int>>& pairs) {
        vector<range> Range;

        // 构造 Range 数组
        for (int i = 0; i < pairs.size(); i++) {
            int a = pairs[i][0], b = pairs[i][1];
            Range.push_back({a, b});
        }

        // 按区间的左端点排序
        sort(Range.begin(), Range.end());

        int n = Range.size();
        int res = 0;
        int r = -2e9; // 初始化右端点为负无穷

        // 遍历区间数组
        for (int i = 0; i < n; i++) {
            // 如果当前区间的左端点大于上一个选择的区间的右端点
            if (Range[i].l > r) {
                res++;  // 可以形成一条链
                r = Range[i].r;  // 更新右端点为当前选择区间的右端点
            } else {
                // 如果无法形成新的链，选择右端点更小的区间
                r = min(r, Range[i].r);
            }
        }

        cout << res;
        return res;
    }
};


class Solution {
public:
    struct range {
        int l, r;
        bool operator<(const range &R) const {
            return l < R.l;  // 按照左端点排序
        }
    };
    static const int N=1010;
    int f[N];
    int findLongestChain(vector<vector<int>>& pairs) {
        vector<range> Range;

        // 构造 Range 数组
        for (int i = 0; i < pairs.size(); i++) {
            int a = pairs[i][0], b = pairs[i][1];
            Range.push_back({a, b});
        }
        
        // 按区间的左端点排序
        sort(Range.begin(), Range.end());
        int n=pairs.size();
        //利用LIS来做
        //枚举以第i个区间结尾
        for(int i=0;i<n;i++){
            f[i]=1;
            for(int j=0;j<i;j++){
                if(Range[i].l>Range[j].r)
                f[i]=max(f[i],f[j]+1);
            }
        }

        return ranges::max(f);
    }
};
```

##### [435. 无重叠区间](https://leetcode.cn/problems/non-overlapping-intervals/)

> 思路：采用贪心做法，将区间按照左端点进行排序，每次只有当当前区间的左端点大于上个区间的右端点，才能够选择，选择后，更新区间的右端点。最后得到的答案为最长的不重叠的区间，求的为删除区间的最小次数，用n-res
>
> ### 关键点
>
> 1. 本题不能使用DP，由于n为1e5

```c++
class Solution {
public:
    static bool cmp(const vector<int>& r1, const vector<int>& r2) {
        return r1[1] < r2[1];  // 按右端点排序
    }

    int eraseOverlapIntervals(vector<vector<int>>& ranges) {
        int n = ranges.size();
        if (n == 0) return 0;

        // 按照右端点排序
        sort(ranges.begin(), ranges.end(), cmp);

        int res = 0;
        int r = -0x3f3f3f3f;  // 初始化右端点为负无穷

        for (int i = 0; i < n; i++) {
            if (ranges[i][0] >= r) {
                res++;
                // 当前区间的左端点大于等于上一个区间的右端点，不重叠
                r = ranges[i][1];  // 更新右端点为当前区间的右端点
            }
        }
        return n-res;
    }
};

```

##### [1546. 和为目标值且不重叠的非空子数组的最大数目](https://leetcode.cn/problems/maximum-number-of-non-overlapping-subarrays-with-sum-equals-target/)

> 思路：本题，要找最多的不重叠子区间，同时子区间和要等于`target`。我们可以使用前缀和+哈希表的形式来做。利用前缀和，不断的往后进行枚举区间的右端点，判断前面是否有区间的和等价与`sr-s(l-1)=target`—–>`s(l-1)=sr-target`。哈希表中存放的是区间和为`sum`的右端点。我们根据区间和公式，获取到和为`sr-target`区间的右端点，将右端点+1得到当前区间的左端点`left`。如果有`left>last_end`则说明，当前选择区间跟上一个选择区间不重叠。加入计数。`last_end`表示上一个区间的右端点。最后需要更新前缀和区间的右端点i
>
> 思路2：可以将上面拆分成一个数组来存储区间，这样就变成了找最多的不重叠区间了

```c++
class Solution {
public:
    int maxNonOverlapping(vector<int>& nums, int target) {
        unordered_map<int, int> presum_map; // 存储前缀和及其对应的下标
        int prefix_sum = 0;
        int count = 0;
        int last_end = -1; // 上一个有效区间的结束下标

        presum_map[0] = -1; // 处理前缀和为target的情况

        for (int i = 0; i < nums.size(); i++) {
            prefix_sum += nums[i];
            // 检查是否存在 prefix_sum - target
            if (presum_map.count(prefix_sum - target)) {
                int start_index = presum_map[prefix_sum - target];
                // 确保新区间不与之前的区间重叠即  当前区间左端点一定要>上一个选择区间的右端点
                if (start_index+1>last_end) {
                    count++;
                    last_end = i; // 更新结束下标
                }
            }

            // 更新当前前缀和的最小下标
            presum_map[prefix_sum] = i;
        }

        return count;
    }
};

#include <vector>
#include <unordered_map>
#include <algorithm>

using namespace std;

class Solution {
public:
    int maxNonOverlapping(vector<int>& nums, int target) {
        unordered_map<int, int> prefixSumMap; // 存储前缀和及其最后出现的位置
        int prefixSum = 0;
        vector<pair<int, int>> intervals; // 存储符合条件的区间
        
        // 计算前缀和
        for (int i = 0; i < nums.size(); i++) {
            prefixSum += nums[i];
            // 检查前缀和减去目标是否存在
            if (prefixSum == target) {
                intervals.emplace_back(0, i); // 从0到i的区间
            }
            if (prefixSumMap.count(prefixSum - target)) {
                intervals.emplace_back(prefixSumMap[prefixSum - target] + 1, i); // 更新区间
            }
            prefixSumMap[prefixSum] = i; // 更新前缀和及其索引
        }

        // 按照右端点排序
        sort(intervals.begin(), intervals.end(), [](const pair<int, int>& a, const pair<int, int>& b) {
            return a.second < b.second; // 按右端点升序排序
        });

        int count = 0;
        int end = -1; // 记录上一个选择的区间的右端点
        
        // 选择不重叠的区间
        for (const auto& interval : intervals) {
            if (interval.first > end) { // 如果当前区间不重叠
                count++;
                end = interval.second; // 更新右端点
            }
        }

        return count; // 返回不重叠区间的最大数量
    }
};
```



#### 区间分组

#### 区间选点

#### 区间覆盖

#### 区间合并





## 排序不等式贪心

### [1402. 做菜顺序](https://leetcode.cn/problems/reducing-dishes/)

> 思路：利用排序贪心，此题式子类似`a[0]⋅b[0]+a[1]⋅b[1]+⋯+a[n−1]⋅b[n−1]` 因此我们需要最大化这个式子，那么需要将a从小到大排序，将b从小到大。此时可以得到。对于本题来说，我们将数组从小到大排序，然后倒着便利，累加和，同时不断的将和累加到res结果中。如果到了某个时候，和小于0了，说明如果继续往下遍历，会让结果越来越小。因此直接退出即可。
>
> #### 关键点
>
> 1. 本题实质上，就是在求最大的累加前缀和。

```c++
class Solution {
public:
    int maxSatisfaction(vector<int>& nums) {
        sort(nums.begin(), nums.end());  // 排序

        int res = 0;
        int count = 1;  // 计数器
        int len = nums.size();  // 数组的长度
        int total = 0;  // 用来记录当前的和

        // 从后往前遍历，因为前面是小的负数，后面是大的正数
        for (int i = len - 1; i >= 0; i--) {
            total += nums[i];  // 将当前元素加入total中
            if (total < 0) {
                break;  // 如果累加到负数，则停止，因为包含更多负数只会让结果更差
            }
            res += total;  // 将total累加到res中
        }

        return res;
    }
};
```







