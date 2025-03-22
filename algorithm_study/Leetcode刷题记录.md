1. # Leetcode刷题记录

## 1.hash表

### hot1两数之和

> 解法一：暴力，时间过高
>
> 解法二：利用hash表简化循环次数。
>
> 具体过程：
>
> 1. 遍历数组nums，判断map中是否存在key为taget-nums[i]的键值对，如果有，找到，如果不存在则将nums[i]为键,数组下标为值存入map中

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        map<int,int> nums_map;
        vector<int> res(2);
        bool flag=false;
        for(int i=0;i<nums.size()&&!flag;i++){
            if(!nums_map.count(target-nums[i]))
                nums_map.insert(make_pair(nums[i],i));
            else{
                flag=true;
                res[0]=nums_map.find(target-nums[i])->second;
                res[1]=i;
            }
        }
        return res;
    }
};
```

### hot2最长连续序列

> 解法：
>
> 1. 由于给定数组可能有重复值，因此利用hash_set先去重，
> 2. 判断set中是否存在nums[i]-1，如果不存在则代表当前数为序列开头，循环判断set中是否存在nums+1 存在序列长度+1，不存在跳出循环
> 3. 与最初定义返回值比较，大的赋给返回值

```c++
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        //定义一个set去重
        unordered_set<int> num_set(nums.begin(),nums.end());
        int res=0;
        int loc=1;
        /**
        每次判断该位置的数是否为起始位置，迭代loc
        **/
        for(int i=0;i<nums.size();i++){
            //如果当前数为开头
            if(!num_set.count(nums[i]-1)){
                int temp=nums[i];
                loc=1;
            //统计以该数为起点序列的长度
            while(num_set.count(++temp))
                loc++;
                if(loc>res)
                res=loc;
            }
        }
        return res;
    }
};
```

### hot3字母异位词分组

> 思路：由于字母异位词所包含字符相同，仅仅顺序不一致，维护一个以排序过后异位词为key，vector<string> 为value的hashmap。即Map<`sort(word)`,`vector<string>`>
>
> 解法：
>
> 1. 遍历字符串数组strs，将每个字符串排序，存入hash表中

```c++
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
       /**
        利用hash表来实现，利用单词排序相同的特征。
        建立一个hash表，以单词排序后的结果为键，数组为值
        **/
        map<string,vector<string>> my_map;
        for(int i=0;i<strs.size();i++){
            //取出单词
            string temp=strs[i];
            //排序获得新单词
            sort(temp.begin(),temp.end());
            //检查hash表中是否有改序列
            my_map[temp].push_back(strs[i]);
        }
        vector<vector<string>> res;
        for(auto pair:my_map){
            res.push_back(pair.second);
        }
        return res;
    }
};
```

## 2.双指针

### hot4移动零

> 思路：利用快排思想，用双指针，left，right。right指针向右移动，不为零与left交换，同时left+1，right+1，为零时仅right+1

```c++
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int left=0;int right=0;
        while(right<nums.size()){
            if(nums[right]!=0){
                swap(nums[left],nums[right]);
                left++;
            }
            right++;    
        }
    }
};
```

### hot5盛最多水容器

> 思路：利用双指针，两指针left，right往中间移动，每次计算更行最大水量
>
> 解法：
>
> 1. 遍历数组`height`，`left`指针位于开头，`right`指针位于末尾
> 2. 计算水量s，移动小的一个指针

```c++
class Solution {
public:

    int maxArea(vector<int>& height) {
        int maxres=0;
        int left=0;int right=height.size()-1;
        while(left<right){
            int area=(right-left)*min(height[left],height[right]);
            maxres=max(maxres,area);
            if(height[left]>=height[right])
                right--;
            else
                left++;
        }
        return maxres;
    }
};
```

### hot6三数之和

> 思路1：利用排序+双指针，同时维护一个指针k，指针k代表数组最左元素，向末尾移动（注意去重），i，j指针代表区间（k+1,len(nums)）。每次计算`s=nums[k]+nums[i]+nums[j]`。根据s的大小判断具体移动哪个指针（i，j）。若`s>0`移动j指针，同时注意去重，若`s<0`移动i指针，注意去重。`s==0`添加到`vector`中。同时移动i，j指针。注意去重。
>
> 解法1：
>
> 1. 当 nums[k] > 0 时直接break跳出：因为 nums[j] >= nums[i] >= nums[k] > 0，即 333 个元素都大于 000 ，在此固定指针 k 之后不可能再找到结果了。
>
> 2. 当 k > 0且nums[k] == nums[k - 1]时即跳过此元素nums[k]：因为已经将 nums[k - 1] 的所有组合加入到结果中，本次双指针搜索只会得到重复组合。
>
> 3. i，j 分设在数组索引 (k,len(nums))(k, len(nums))(k,len(nums)) 两端，当i < j时循环计算s = nums[k] + nums[i] + nums[j]，并按照以下规则执行双指针移动：
>
>    * 当s < 0时，i += 1并跳过所有重复的nums[i]；
>    * 当s > 0时，j -= 1并跳过所有重复的nums[j]；
>    * 当s == 0时，记录组合[k, i, j]至res，执行i += 1和j -= 1并跳过所有重复的nums[i]和nums[j]，防止记录到重复组合。
>
>    思路2：利用两重循环+hash记录
>
> 

```c++
//解法1
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        /**
        思想：先将数组排好序，指定三个指针k,i,j，k指向左边第一个，i，j代表区间（K,len(nums)）
        **/
        vector<vector<int>> res;
        int i=1;
        int j=nums.size()-1;
        int k=0;
        //数组排序
        sort(nums.begin(),nums.end());
        for(;k<nums.size();k++){
            if(k>0&&nums[k]==nums[k-1])
            continue;
            for(int i=k+1,j=nums.size()-1;i<j;){
                int s=nums[k]+nums[i]+nums[j];
                if(s==0){
                res.push_back({nums[k],nums[i],nums[j]});
                //移动i,j指针
                 while(i<j&&nums[i]==nums[++i]);
                 while(i<j&&nums[j--]==nums[j]);
                }
                else if(s>0){
                    //说明nums[j]太大了，左移动j指针
                    while(i<j&&nums[j--]==nums[j]);
                }else{
                     //说明nums[i]太小了，右移动i指针
                     while(i<j&&nums[i]==nums[++i]);
                }
            }

        }
        return res;
    }
};
//解法2class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> res;
        sort(nums.begin(),nums.end());
        for(int i=0;i<nums.size();i++){
            //确定一个数字
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            unordered_set<int> my_set{};
            int one=nums[i];
            for(int j=i+1;j<nums.size();j++){
                int two=nums[j];
                if(!my_set.count(-(one+two)))
                    my_set.insert(two);
                else{
                    res.push_back({one,two,-(one+two)});
                    while (j + 1 < nums.size() && nums[j] == nums[j + 1]) ++j;
                }
            }
        }
        return res;
    }
};
```

### [16. 最接近的三数之和](https://leetcode.cn/problems/3sum-closest/)

> 思路：采用排序过后使用相向双指针的思路，同时利用一个`sum`变量记录最终答案，使用一个`temp_sum`记录当前三个指针所指向的总和值，采用一个`sub`变量，代表与`target`的距离。
>
> 解法：
>
> 1. **排序**：首先对输入数组 `nums` 进行排序，以便使用双指针方法。
>
> 2. **初始化变量**：
>
>    - `sum`：用于存储最接近目标值的三数之和。
>
>    - `temp_sum`：用于存储当前计算的三数之和。
>
>    - `sub`：用于存储当前最小的差值，初始值为 `INT_MAX`。
>
> 3. **三重循环**：
>
>    - 外层循环：遍历数组，选择第一个数 `x`。
>
>    - 内层循环：使用双指针方法，指针 `j` 从 `i+1` 开始，指针 `k` 从数组末尾开始。
>
> 4. **计算三数之和**：
>
>    - 计算当前三数之和 `temp_sum = x + nums[j] + nums[k]`。
>
>    - 计算 `temp_sum` 与 `target` 的差值 `temp = abs(target - temp_sum)`。
>
> 5. **更新最小差值和结果**：
>    - 如果当前差值 `temp` 小于之前记录的最小差值 `sub`，则更新 `sub` 和 `sum`。
>
> 6. **移动指针**：
>
>    - 如果当前三数之和小于目标值 `target`，移动左指针 `j++` 以增大和。
>
>    - 如果当前三数之和大于目标值 `target`，移动右指针 `k--` 以减小和。
>
>    - 如果当前三数之和等于目标值 `target`，直接返回目标值。
>
> 7. **返回结果**：遍历完所有可能的组合后，返回 `sum`，即最接近目标值的三数之和。

```c++
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        int sum = 0, temp_sum = 0;
        int j, k;
        int sub = INT_MAX; // 用于存储当前最小的差值
        sort(nums.begin(), nums.end()); // 先对数组进行排序
        int n = nums.size();
        
        for (int i = 0; i < n - 2; i++) {
            int x = nums[i];
            j = i + 1;
            k = n - 1;

            while (j < k) {
                temp_sum = x + nums[j] + nums[k]; // 计算三个数的和
                int temp = abs(target - temp_sum); // 计算当前和与目标值的差值
                
                if (temp < sub) { // 如果当前差值更小，更新sub和sum
                    sub = temp;
                    sum = temp_sum;
                }
                if (temp_sum < target) // 如果当前和小于目标值，移动左指针
                    j++;
                else if (temp_sum > target) // 如果当前和大于目标值，移动右指针
                    k--;
                else // 如果当前和等于目标值，直接返回目标值
                    return target;
            }
        }
        return sum; // 返回最接近目标值的三数之和
    }
};

```

### [18. 四数之和](https://leetcode.cn/problems/4sum/)

> 思路：与三数之和类似，只不过多了一层循环，同时需要注意的点在于，需要对前两个数，判重。`if (i > 0 && nums[i] == nums[i - 1]) continue; `以及`if (j > i + 1 && nums[j] == nums[j - 1]) continue;`如果符合条件，说明在之前就已经统计过该值了，需要跳过。
>
> 优化：
>
> 1. 如果当前组合的最小和都大于目标值，直接跳出循环
>
>    ```c++
>    if ((long)nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target) {
>        break;  // 如果当前组合的最小和都大于目标值，直接跳出循环
>    }
>    ```
>
> 2. 如果最后几个数加起来都小于目标值，直接跳出循环。
>
>    ```c++
>    if ((long)nums[i] + nums[n - 3] + nums[n - 2] + nums[n - 1] < target) {
>        continue;  // 如果当前组合的最大和都小于目标值，跳过当前组合
>    }
>    ```
>
>    解法：
>
> 1. **排序数组**：首先将输入数组排序，这有助于后续使用双指针法。
>
> 2. **外层循环**：固定第一个数，==跳过重复的元素==，避免结果中出现重复的四元组。
>
> 3. **提前判断**：通过最小和最大和判断，提前跳出不必要的循环，减少计算量。
>
> 4. **中层循环**：固定第二个数，==同样跳过重复的元素==。
>
> 5. **双指针法**：内层循环中使用双指针法，左右指针分别从当前固定元素之后的数组部分开始和结束处向中间移动，寻找符合条件的两数之和。
>
> 6. **跳过重复元素**：在找到符合条件的四元组后，继续移动指针，跳过重复的元素，确保结果唯一性。
>
> 7. **返回结果**：所有符合条件的四元组存入结果数组并返回。

```c++
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        vector<vector<int>> res;  // 用于存储结果的二维向量
        sort(nums.begin(), nums.end());  // 对数组进行排序
        int n = nums.size();
        if (n < 4) return res;  // 如果数组长度小于4，直接返回空结果
        
        // 外层循环，固定第一个数
        for (int i = 0; i < n - 3; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;  // 跳过重复的元素
            
            // 通过提前判断，减少不必要的计算
            if ((long)nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target) {
                break;  // 如果当前组合的最小和都大于目标值，直接跳出循环
            }
            if ((long)nums[i] + nums[n - 3] + nums[n - 2] + nums[n - 1] < target) {
                continue;  // 如果当前组合的最大和都小于目标值，跳过当前组合
            }
            
            // 中层循环，固定第二个数
            for (int j = i + 1; j < n - 2; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1]) continue;  // 跳过重复的元素
                
                int k = j + 1, l = n - 1;  // 双指针初始化
                
                // 内层循环，使用双指针法寻找符合条件的两数之和
                while (k < l) {
                    long long sum = (long long)nums[i] + nums[j] + nums[k] + nums[l];  // 计算四数之和
                    
                    if (sum > target) {
                        l--;  // 如果和大于目标，右指针左移
                    } else if (sum < target) {
                        k++;  // 如果和小于目标，左指针右移
                    } else {
                        res.push_back({nums[i], nums[j], nums[k], nums[l]});  // 找到一个符合条件的四元组
                        
                        // 跳过重复的元素
                        while (k < l && nums[k] == nums[k + 1]) k++;
                        while (k < l && nums[l] == nums[l - 1]) l--;
                        
                        k++;  // 更新指针位置
                        l--;
                    }
                }
            }
        }
        
        return res;  // 返回结果
    }
};

```



## 3.链表

### hot160相交链表

> 思路1：分别求两链表的长度，再求长度之差，对齐链表。再进行遍历
>
> :star: 思路2：利用两指针消除长度差，若相交，链表A： a+c, 链表B : b+c. a+c+b+c = b+c+a+c 。则会在公共处c起点相遇。若不相交，a +b = b+a 。因此相遇处是NULL
>
> 解法2：
>
> 1、定义A,B两个指针，分别遍历A，B链表。当A指针走到终点NULL时赋值 A=headB。B指针走到终点NULL时，赋值B=headA。
>
> 2、二者相遇点即为公共处，或者为NULL。

```c++
//解法1
class Solution {
public:
    ListNode* getIntersectionNode(ListNode* headA, ListNode* headB) {
        int lenA = 0;
        ListNode* tempA = headA;
        int lenB = 0;
        ListNode* tempB = headB;
        while (tempA != NULL) {
            lenA++;
            tempA = tempA->next;
        }
        while (tempB != NULL) {
            lenB++;
            tempB = tempB->next;
        }
        int step;
        if (lenA >= lenB) {
            step = lenA - lenB;
            while (step) {
                headA = headA->next;
                step--;
            }
        } else {
            step = lenB - lenA;
            while (step) {
                headB = headB->next;
                step--;
            }
        }
        while (headA != NULL && headB != NULL) {
            if (headA == headB)
                return headA;
                headA = headA->next;
                headB = headB->next;
        }
        return NULL;
    }
};

//解法2
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode *PA=headA;
        ListNode *PB=headB;
        while(PA!=PB){
            if(PA==NULL){
                //A链表走到尾部
                PA=headB;
            }else{
                PA=PA->next;
            }
            if(PB==NULL){
                //B链表走到尾部
                PB=headA;
            }else{
                PB=PB->next;
            }
        }
        return PA;
    }
};
```



### hot206反转链表

> 思路1：普通的双指针迭代。
>
> 思路2：递归
>
> 解法1：
>
> 1、定义pre指针为NULL，代表头结点前一个节点NULL。current指针为head。以及保存current下一个节点的指针newNext。
>
> 解法2：
>
> 1、重点为，递归到末尾要返回新链表的头结点。同时改变前后指针指向。

```c++
//解法1:双指针迭代
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode *current=head;
        ListNode *pre=NULL;
        ListNode *newNext;
        while(current!=NULL){
            newNext=current->next;
            current->next=pre;
            pre=current;
            current=newNext;
        }
        return pre;
    }
};

//解法2：递归
class Solution {
public:

    ListNode * recur(ListNode *cur,ListNode *pre){
        if(cur==NULL)
            return pre;
          ListNode *res=recur(cur->next,cur);//返回反转后头结点
            cur->next=pre;
            return res;
    }
    ListNode* reverseList(ListNode* head) {
        return recur(head,NULL);
    }
};
```

### 反转链表二

> 思路1：创建哑节点、移动指针到指定位置、反转链表指定区域，然后将反转区域重新连接到链表的其他部分来实现链表的部分反转。
>
> 解法1：
>
> 1. 创建一个哑节点 `dump` 并将其 `next` 指针指向 `head`，以处理头节点的边界情况。
>
> 2. 通过循环将 `p` 和 `q` 分别移动到 `left` 和 `right` 的位置。`pre` 指针用于记录 `left` 前一个节点。
>
> 3. 将 `p` 和 `pre->next` 的连接断开。
>
>    通过循环将 `left` 到 `right` 区间的节点逐步插入到 `pre->next`，从而实现局部反转。
>
> 4. 将反转区域的最后一个节点的 `next` 指针连接到原链表的 `end` 节点。
>
>    将 `pre->next` 重新连接到反转后的链表头节点。
>
> 思路2：直接调整指针的指向，不用断开链表。总共三个指针
>
> 1. `curr`：指向待反转区域的第一个节点 `left`；
> 2. `next`：永远指向 `curr `的下一个节点，循环过程中，`curr `变化以后 `next `会变化；
> 3. `pre`：永远指向待反转区域的第一个节点 `left `的前一个节点，在循环过程中不变
>
> 解法2：
>
> 1. 创建一个哑节点 `dump`，并将其 `next` 指针指向 `head`。这有助于处理链表头的边界情况。
> 2. 通过循环，将 `pre` 移动到 `left` 的前一个节点。
> 3. 循环`right-left`次，`cur`节点只向`next`的下一个节点.`next`指针指向最左边`left`边界的节点。`pre`指针指向`next`节点。也就是`pre->next`，

```c++
//解法1
class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        ListNode *pre, *end;
        ListNode *p, *q;
        ListNode* dump = new ListNode(-1, head); // 创建一个哑节点
        p = dump;
        q = dump;
        // 移动p，q到对应为止
        while (left--) {
            pre = p;
            p = p->next;
        }
        while (right--) {
            q = q->next;
        }
        end = q ? q->next : nullptr;
        ListNode* temp;
        pre->next = nullptr;
        ListNode *newHead=nullptr;
        while (p != end) {
            temp = p->next;
            p->next = pre->next;
            pre->next = p;
            newHead=p;
            p = temp;
        }
        pre->next = newHead;
        while(newHead->next!=nullptr)
            newHead=newHead->next;
        newHead->next=end;
        return dump->next;
    }
};

//解法2
class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        ListNode *dump=new ListNode(-1,head);
        ListNode *pre=dump,*cur,*nextNode;
        for(int i=1;i<left;i++){
            pre=pre->next;
        }
        cur=pre->next;
        for(int i=0;i<right-left;i++){
            nextNode=cur->next;
            cur->next=nextNode->next;
            nextNode->next=pre->next;
            pre->next=nextNode;
        }
        return dump->next;
    }
};
```



### hot234回文链表

> 思路1：利用数组将链表的值存起来，之后利用双指针（`left，right`）向中间遍历验证。
>
> 思路2：利用双指针+反转链表。slow指针每次走一步，fast指针每次走两步。当fast指针走到末尾时，slow指针指向链表终点.之后反转后半部分链表。

```c++
//解法2
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        if (head == NULL)
            return true;
        
        ListNode *slow = head;
        ListNode *fast = head;
        
        // 找到链表的中点
        while (fast != NULL && fast->next != NULL) {
            slow = slow->next;
            fast = fast->next->next;
        }
        
        // 反转链表的后半部分
        ListNode *Pre = NULL;
        ListNode *R;
        while (slow) {
            R = slow->next;
            slow->next = Pre;
            Pre = slow;
            slow = R;
        }
        
        // 比较前半部分和反转后的后半部分
        ListNode *begin = head;
        while (Pre) {
            if (Pre->val != begin->val)
                return false;
            Pre = Pre->next;
            begin = begin->next;
        }
        
        return true;
    }
};
```

### hot141环形链表

> 思路1：采用hast_set记录访问过的链表节点
>
> 思路2：利用快慢指针。如果有环，快指针一定可以追上慢指针。

```c++
//解法2
class Solution {
public:
    bool hasCycle(ListNode* head) {
        if (head == nullptr || head->next == nullptr) {
            return false;
        }
        ListNode* slow = head;
        ListNode* fast = head->next;
        while (slow != fast) {
            if (fast == nullptr || fast->next == nullptr) {
                return false;
            }
            slow = slow->next;
            fast = fast->next->next;
        }
        return true;
    }
};
```

### hot环形链表二

> 思路1：利用hash_set
>
> 思路2：利用快慢指针。同时注意:
>
> 1. 第一次相遇，快慢指针在环内部相遇。
>
> 2. 第二次相遇，快慢指针在环入口相遇。
>
> ​	解释：这段代码通过两个阶段（检测环和找到环的起点）
>
> * 检测环：由于`fast`指针每次移动两步，而`slow`指针每次移动一步，因此如果链表中存在环，两个指针最终会在环中相遇。这个相遇点可以证明链表中存在环。
> * 找到环：`fast` 比 `slow` 多走了*n* 个环的长度，即 f=s+nb 当前`fast`以及`slow`分别走了2n和n个环的周长。而每次经过入口处都走了`a+nb`的距离，因此fast以及slow都需要在走`a`的距离。a的距离就表示从`head`到入口的距离。因此仅需将fast赋值为head。当fast==slow时。二者相遇

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        if(head==NULL||head->next==NULL)
            return NULL;
        ListNode *slow=head;
        ListNode *fast=head;
        while(true){
            if(fast==NULL||fast->next==NULL)
                return NULL;
            //第一次相遇
            slow=slow->next;
            fast=fast->next->next;
            if(slow==fast)
                break;   
        }
        // if(slow==NULL||fast==NULL)
        //     return NULL;
        //此时slow=nb,还需要走a步到达相遇点
        fast=head;
        while(fast!=slow){
            fast=fast->next;
            slow=slow->next;
        }
        return slow;
    }
};
```

### hot合并两个有序链表

> 思路：创建头节点，采用尾差法

```c++
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode dummy(0);
        ListNode *head;
        head=&dummy;
        ListNode *PA=list1;
        ListNode *PB=list2;
        while(PA&&PB){
            if(PA->val>=PB->val){
                head->next=PB;
                PB=PB->next;
            }else{
                head->next=PA;
                PA=PA->next;
            }
            head=head->next;
        }
        head->next=PA?PA:PB;
        return dummy.next;
       }
};
```

### hot两数相加

> 思路1：将两链表的和用数字表示出来，再依次创建新的链表节点。（链表太长，无法使用数据类型装下）
>
> 思路2：利用加法原则，每加一次，生成一个链表节点。
>
> 解法：
>
> 1. 创建一个哑节点，避免讨论边界情况。
> 2. 创建两个指针`A,B`，分别指向`l1,l2`链表。A，B指针往后遍历。考虑`A+B+carry`，`carry`为进位。主要分为以下三种情况。
> 3. 1、A不为空，B不为空，则产生新节点的值为`val=(A+B+carry)%10`。同时新的`carry`进位为`(A+B+carry)/10`
> 4. 2、A不为空，B为空，则产生新节点的值为`val=(A+0+carry)%10`。同时新的`carry`进位为`(A+0+carry)/10`
> 5. 3、A为空，B不为空，则产生新节点的值为`val=(B+0+carry)%10`。同时新的`carry`进位为`(B+0+carry)/10`
> 6. 最后考虑如果`carry`不为0，那么还需要新创建一个节点。

```c++
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
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        // Create a dummy head node.
        ListNode* dummyHead = new ListNode(-1);
        ListNode *current = dummyHead;
        int carry = 0;
        
        while (l1 || l2 || carry) {
            int sum = carry;
            if (l1) {
                sum += l1->val;
                l1 = l1->next;
            }
            
            if (l2) {
                sum += l2->val;
                l2 = l2->next;
            }
            
            carry = sum / 10;
            ListNode *newNode = new ListNode(sum % 10);
            current->next = newNode;
            current = current->next;
        }
        
        return dummyHead->next;
    }
};

```

### hot删除链表倒数第n个节点

> **:smile: 注意**：对于链表问题，如果没给哑节点，最好手动给出哑节点。可以避免讨论边界问题。
>
> 思路1：计算链表的长度，指针应该移动到被删除节点的前一个位置。移动次数sub=len-n+1，此为没有加上哑节点的sub，如果加上哑节点。sub=len-sub
>
> 思路2：快慢指针。由于需要删除倒数第`n`个节点。可以考虑采用快慢指针。快慢指针间隔`n`个节点。当`fast`指针移动到末尾`NULL`时。慢指针刚好移动到被删除节点的前一个位置。

```c++
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
//快慢指针法
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        //创建一个哑节点，避免讨论。
        ListNode *dump=new ListNode(-1,head);
        ListNode *slow=dump;
        ListNode *fast=head;
        while(n--){
            fast=fast->next;
        }
        while(fast){
            slow=slow->next;
            fast=fast->next;
        }
        slow->next=slow->next->next;
        return dump->next;
    }
};
```

### hot两两交换链表中节点

> 思路1：迭代，用两个指针`left`，`right`来进行指针的交换。定义一个`pre`指针，指向`left`的前一个节点。同时，增加一个哑节点。终止条件：`left`和`right`之中有一个指针为`NULL`，最后返回哑节点下一个节点。
>
> 思路2：递归。
>
> 解法：
>
> 1. 终止条件：当链表仅有一个节点或为空时返回
> 2. 函数内部交换节点，`head`和`nextNode`。head指向返回的链表节点。`nextNode`指向head。同时，交换完成后，返回交换完成节点的头结点。即`nextNode`

```c++
//思路1:迭代
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        ListNode *dummy = new ListNode(-1, head); // 定义一个哑节点
        ListNode *pre = dummy;                    // 指向哑节点
        ListNode *left = head;                    // 指向当前节点
        ListNode *right = head ? head->next : nullptr; // 指向下一个节点

        while (left != nullptr && right != nullptr) {
            pre->next = right;            // 前一个节点的next指向右节点
            left->next = right->next;     // 左节点的next指向右节点的next
            right->next = left;           // 右节点的next指向左节点

            pre = left;                   // 前一个节点移动到左节点
            left = left->next;            // 左节点移动到下一个节点
            right = left ? left->next : nullptr; // 右节点移动到下一个节点的下一个节点
        }

        ListNode *newHead = dummy->next;  // 新的头节点是哑节点的next
        delete dummy;                     // 删除哑节点
        return newHead;                   // 返回新的头节点
    }
};
//解法2:递归
class Solution {
public:
    // 使用递归方法实现链表中节点的两两交换

    // 函数swapPairs接收链表头节点head作为参数
    ListNode* swapPairs(ListNode* head) {
        // 递归终止条件：如果链表为空或只有一个元素，无需交换，直接返回原链表头
        if(head == NULL || head->next == NULL)
            return head;

        // nextNode 指向当前待交换对的第一个节点的下一个节点
        ListNode *nextNode = head->next;
     
        // 对后面的链表进行递归调用，返回值作为当前head节点的新next（即交换后的第二个节点的新next）
        head->next = swapPairs(nextNode->next);
        
        // 完成一次交换：nextNode（原第二个节点）的next指向当前head（原第一个节点）
        nextNode->next = head;
        
        // 返回交换后的子链表头节点，即原来链表中的第二个节点
        return nextNode;
    }
};
```

### 随机链表的复制

> 思路1：利用`map`建立原节点与新创建节点的映射关系。
>
> 思路2：在原链表后面复制新节点

```c++
/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* next;
    Node* random;
    
    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};
*/

class Solution {
public:
    Node* copyRandomList(Node* head) {
        map<Node*,Node *> my_map;
        Node *p=head;
        Node *dump=new Node(-1);
        Node *chead=dump;
        while(p!=NULL){
            Node *newNode=new Node(p->val);
            my_map[p] = newNode;
            p=p->next;
        }
        p=head;
        chead=dump;
        while(p!=NULL){
            chead->next=my_map[p];
            my_map[p]->random=my_map[p->random];
            p=p->next;
            chead=chead->next;
        }
         return dump->next;
    }
   
};
//直接在原链表节点后面复制新节点
/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* next;
    Node* random;
    
    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};
*/

class Solution {
public:
    Node* copyRandomList(Node* head) {
        if (head == NULL) {
            return NULL;
        }

        // 第一步：在每个节点后面创建一个新节点
        Node* p = head;
        while (p != NULL) {
            Node* newNode = new Node(p->val);
            newNode->next = p->next;
            p->next = newNode;
            p = newNode->next;
        }

        // 第二步：设置新节点的random指针
        p = head;
        while (p != NULL) {
            Node* copyNode = p->next;
            if (p->random != NULL) {
                copyNode->random = p->random->next;
            }
            p = copyNode->next;
        }

        // 第三步：拆分链表，将新旧节点分开
        p = head;
        Node* newHead = head->next;
        Node* q = newHead;
        while (p != NULL) {
            p->next = q->next;
            p = p->next;
            if (p != NULL) {
                q->next = p->next;
                q = q->next;
            }
        }

        return newHead;
    }
};
```

### hot链表排序（:joy_cat:不会 ）

> 思路1：采用归并排序。

```c++
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
/**思路：1、如果链表为空或者仅有一个元素。返回
        2、获取链表长度，开始归并排序，排序间隔为1、2、4、8一直到n
        3、归并时，先拆分链表，再合并。
**/
    ListNode* sortList(ListNode* head) {
            if(!head||!head->next)
                return head;
            int len=0;
            ListNode *current=head;
            while(current){
                len++;
                current=current->next;
            }
            //定义三个指针，同时创建哑节点。
            ListNode *dump=new ListNode(-1,head);
            ListNode *pre=dump;
            ListNode *left;
            ListNode *right;
            ListNode *tail;
            //主要部分，开始拆分合并
            for(int step=1;step<len;step*=2){
                pre=dump;
                current=dump->next;
                while(current){
                    left=current;//第一个节点
                    right=split(current,step);//第二个节点。
                    current=split(right,step);//剩余链表的头结点
                    tail=merge(pre,left,right);//合并以left，right开头的链表。
                    pre=tail;//合并后链表的尾结点。
                }
            }
            return dump->next;
    }

    ListNode  *split(ListNode *head,int n){
        //切分长度为step的链表
        while(--n&&head){
            head=head->next;
        }
        //定义一个rest指针返回剩余链表头结点。
        ListNode *rest=head?head->next:NULL;
        //将长度为step的链表与剩余链表断开
        if(head)
            head->next=nullptr;
        return rest;
    }
    ListNode * merge(ListNode *pre,ListNode *l1,ListNode *l2){
        ListNode *current=pre;
        while(l1&&l2){
            if(l1->val<=l2->val){
                current->next=l1;
                l1=l1->next;
            }else{
                current->next=l2;
                l2=l2->next;
            }
            current=current->next;
        }
        //将l1,或l2,未链接完的链接到current后面.
        current->next=l1?l1:l2;
        //返回合并后链表尾部
        while(current->next){
            current=current->next;
        }
        return current;
        
    }
};
```

### hotLRU实现

> 思路：利用双向链表+`hashMap`实现

```c ++
struct DLinkedNode {
    int key, value;
    DLinkedNode* prev;
    DLinkedNode* next;
    DLinkedNode(): key(0), value(0), prev(nullptr), next(nullptr) {}
    DLinkedNode(int _key, int _value): key(_key), value(_value), prev(nullptr), next(nullptr) {}
};

class LRUCache {
private:
    unordered_map<int, DLinkedNode*> cache;
    DLinkedNode* head;
    DLinkedNode* tail;
    int size;
    int capacity;

public:
    LRUCache(int _capacity): capacity(_capacity), size(0) {
        // 使用伪头部和伪尾部节点
        head = new DLinkedNode();
        tail = new DLinkedNode();
        head->next = tail;
        tail->prev = head;
    }
    
    int get(int key) {
        if (!cache.count(key)) {
            return -1;
        }
        // 如果 key 存在，先通过哈希表定位，再移到头部
        DLinkedNode* node = cache[key];
        moveToHead(node);
        return node->value;
    }
    
    void put(int key, int value) {
        if (!cache.count(key)) {
            // 如果 key 不存在，创建一个新的节点
            DLinkedNode* node = new DLinkedNode(key, value);
            // 添加进哈希表
            cache[key] = node;
            // 添加至双向链表的头部
            addToHead(node);
            ++size;
            if (size > capacity) {
                // 如果超出容量，删除双向链表的尾部节点
                DLinkedNode* removed = removeTail();
                // 删除哈希表中对应的项
                cache.erase(removed->key);
                // 防止内存泄漏
                delete removed;
                --size;
            }
        }
        else {
            // 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
            DLinkedNode* node = cache[key];
            node->value = value;
            moveToHead(node);
        }
    }

    void addToHead(DLinkedNode* node) {
        node->prev = head;
        node->next = head->next;
        head->next->prev = node;
        head->next = node;
    }
    
    void removeNode(DLinkedNode* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }

    void moveToHead(DLinkedNode* node) {
        removeNode(node);
        addToHead(node);
    }

    DLinkedNode* removeTail() {
        DLinkedNode* node = tail->prev;
        removeNode(node);
        return node;
    }
};
```

### 旋转链表

> 思路1：将链表转为数组，分别反转`0-len-k-1`，`len-k-len-1`,`0-len-1`
>
> 思路2：将链表合并为循环链表，找到合并完成尾结点之前位置。`pos=len-k%len`.从当前节点断开。返回

```c++
class Solution {
public:
    ListNode* rotateRight(ListNode* head, int k) {
        if (k == 0 || head == nullptr || head->next == nullptr) {
            return head;
        }
        int n = 1;
        ListNode* iter = head;
        while (iter->next != nullptr) {
            iter = iter->next;
            n++;
        }
        int add = n - k % n;
        if (add == n) {
            return head;
        }
        iter->next = head;
        while (add--) {
            iter = iter->next;
        }
        ListNode* ret = iter->next;
        iter->next = nullptr;
        return ret;
    }
};
```

### 奇偶链表

> 思路：分为奇链表和偶链表，最后将偶链表链接到奇链表之后。

```c++
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
    ListNode* oddEvenList(ListNode* head) {
        if (!head || !head->next) return head;

        ListNode *odd = head; // 奇数节点链表的起点
        ListNode *even = head->next; // 偶数节点链表的起点
        ListNode *evenHead = even; // 保存偶数节点链表的头部，以便最后连接

        while (even && even->next) {
            odd->next = even->next;
            odd = odd->next;
            even->next = odd->next;
            even = even->next;
        }

        odd->next = evenHead; // 将奇数节点链表和偶数节点链表连接起来

        return head;
    }
};

```

### 链表组件

> 思路：观察组件存在条件：
>
> 1. 节点的值在数组 `nums`中且节点位于链表起始位置；
> 2. 节点的值在数组 `nums` 中且节点的前一个点不在数组 `nums`中。

```c++
class Solution {
public:
    int numComponents(ListNode* head, vector<int>& nums) {
        int sum=0;
        ListNode *pre=nullptr;
        //vector中没有专门查看元素是否存在的方法。因此创建set，调用set中的count方法
        unordered_set<int> numSet;
        for(int num:nums)
             numSet.emplace(num);
        while(head){
            if(numSet.count(head->val)){
            //判断pre是否在nums中
                if(pre!=nullptr&&!numSet.count(pre->val))
                        sum++;
                else if(pre==nullptr)
                    sum++;
            }
            pre=head;
            head=head->next;
        }
        return sum;
    }
};
```

### 删除链表的中间节点

> 思路：快慢指针

```c++
class Solution {
public:
    ListNode* deleteMiddle(ListNode* head) {
        if(head==nullptr||head->next==nullptr)
            return nullptr;
        ListNode *dump=new ListNode(-1,head);
        ListNode *slow=dump;
        ListNode *fast=dump;
        while(fast&&fast->next!=nullptr&&fast->next->next!=nullptr){
            slow=slow->next;
            fast=fast->next->next;
        }
        slow->next=slow->next->next;
        return head;
    }
};
```

### 重排链表

> 思路：重排过后链表为，前半段结构不变，后半段链表反转之后，交替合并。
>
> 解法：
>
> 1. 利用快慢指针找到中间节点。
> 2. 从中间将链表一分为二。后半段链表进行反转。
> 3. 左右两边链表交替合并

```c++
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
    void reorderList(ListNode* head) {
        // 思路：找到中间节点，将链表分为两部分
        // 反转后半部分链表，然后交替合并两部分链表

        // 使用快慢指针法找到中间节点
        ListNode *slow = head;
        ListNode *fast = head;
        while (fast->next && fast->next->next) {
            slow = slow->next;
            fast = fast->next->next;
        }

        // slow为后半部分链表的起点，从slow的下一个节点开始反转
        ListNode *pre = nullptr;
        ListNode *cur = slow->next;
        ListNode *temp;
        while (cur) {
            temp = cur->next;  // 暂存下一个节点
            cur->next = pre;   // 反转当前节点的指针
            pre = cur;         // pre指针后移
            cur = temp;        // cur指针后移
        }
        // 将前半部分链表与后半部分链表分离
        slow->next = nullptr;

        // 开始交替合并两部分链表
        ListNode *first = head;
        ListNode *second = pre;  // pre为反转后的后半部分链表的头节点
        while (second) {
            ListNode *nextNodeRight = second->next; // 暂存second的下一个节点
            ListNode *nextNodeLeft = first->next;   // 暂存first的下一个节点

            // 插入second到first后面
            second->next = first->next;
            first->next = second;

            // 移动指针
            first = nextNodeLeft;
            second = nextNodeRight;
        }
    }
};

```

## 4.二叉树

### 遍历

#### 二叉树中序遍历（6.24复习）

> 思路1：递归
>
> 思路2：利用栈，先将当前节点入栈，然后遍历当前节点的左子树，左子树遍历完成后，从栈中弹出一个节点，访问它，然后转向它的右子树进行访问。
>
> 思路3：`Morris`遍历，利用线索二叉树思想,根据中序遍历，找到根节点的前驱，建立起线索指针。
>
> 解法3：
>
> 1. **判断当前根节点是否有左子树**：如果当前节点没有左子树，则直接访问当前节点。
>
> 2. **找到前驱节点**：如果当前节点有左子树，则找到该节点的前驱节点，即左子树的最右节点。
>
> 3. **判断当前节点是否已被访问**：
>
>    - 如果前驱节点的右指针为空，表示当前节点未被访问，建立临时线索指针，指向当前节点，并移动到左子树。
>
>    - 如果前驱节点的右指针已经指向当前节点，表示左子树已被访问，恢复树的结构，断开临时线索指针，并访问当前节点。
>
> 4. **移动到右子树**：在处理完左子树后，移动到右子树。

```c++
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        stack<TreeNode *> st;
        vector<int> ve;
        TreeNode *p=root;
        while(p||!st.empty()){
            while(p){
                //节点入栈
            st.push(p);
            p=p->left;
            }
            //左子树访问完毕
            //弹出栈中节点，访问
            p=st.top();
            st.pop();
            ve.push_back(p->val);
            //左子树访问完毕。
            //转向右子树
            p=p->right;
        }
        return ve;
    }
};

//思路2
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        if (root == nullptr)
            return res;
        TreeNode *p1 = root, *p2 = nullptr;
        while (p1) {
            p2 = p1->left;
            // 判断当前根节点是否有左子树
            if (!p2) {
                // 如果没有左子树，直接访问当前节点
                res.push_back(p1->val);
            } else {
                // 找到当前节点的前驱节点
                while (p2->right && p2->right != p1)
                    p2 = p2->right;
                // 判断当前节点是否已被访问
                if (p2->right == nullptr) {
                    // 前驱节点的右指针为空，表示当前节点未被访问
                    // 建立临时线索指针，指向当前节点
                    p2->right = p1;
                    p1 = p1->left;
                    continue;
                } else {
                    // 前驱节点的右指针指向当前节点，表示左子树已被访问
                    // 恢复树的结构，断开临时线索指针
                    p2->right = nullptr;
                    // 访问当前节点
                    res.push_back(p1->val);
                }
            }
            // 移动到右子树
            p1 = p1->right;
        }
        return res;
    }
};
```

#### 二叉树先序遍历

> 思路1：递归
>
> 思路2：利用栈模拟
>
> 思路3：`Morris`遍历，借助`Morris`遍历，利用线索二叉树思想,根据中序遍历，找到根节点的前驱，建立起线索指针。
>
> 解法3：
>
> 1. **判断当前根节点是否有左子树**：如果当前节点没有左子树，则直接访问当前节点。
>
> 2. **找到前驱节点**：如果当前节点有左子树，则找到该节点的前驱节点，即左子树的最右节点。
>
> 3. **判断当前节点是否已被访问**：
>
>    - 如果前驱节点的右指针为空，表示当前节点未被访问，访问当前节点，同时建立临时线索指针，指向当前节点，并移动到左子树。
>
>    - 如果前驱节点的右指针已经指向当前节点，表示左子树已被访问，恢复树的结构，断开临时线索指针。
>
> 4. **移动到右子树**：在处理完左子树后，移动到右子树。

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
class Solution {
public:
    vector<int> preorderTraversal(TreeNode* root) {
        stack<TreeNode *> st;
        vector<int> ve;
        TreeNode *p = root;
        while (p || !st.empty()) {
            while (p) {
                // 节点入栈
                st.push(p);
                ve.push_back(p->val);
                p = p->left;
            }
            // 左子树访问完毕
            // 弹出栈中节点，访问
            p = st.top();
            st.pop();
            // 转向右子树
            p = p->right;
        }
        return ve;
    }
};

class Solution {
public:
    // 采用Morris遍历进行前序遍历
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> res;
        if (root == nullptr)
            return res;
        TreeNode *p1 = root, *p2 = nullptr;
        while (p1) {
            p2 = p1->left;
            // 判断当前根节点是否有左子树
            if (!p2) {
                // 如果没有左子树，直接访问当前节点
                res.push_back(p1->val);
            } else {
                // 找到当前节点的前驱节点
                while (p2->right && p2->right != p1)
                    p2 = p2->right;
                // 判断当前节点是否已被访问
                if (p2->right == nullptr) {
                    // 前驱节点的右指针为空，表示当前节点未被访问
                    // 访问当前节点，并建立临时线索指针
                    res.push_back(p1->val);
                    p2->right = p1;
                    p1 = p1->left;
                    continue;
                } else {
                    // 前驱节点的右指针指向当前节点，表示左子树已被访问
                    // 恢复树的结构，断开临时线索指针
                    p2->right = nullptr;
                }
            }
            // 移动到右子树
            p1 = p1->right;
        }
        return res;
    }
};


```

#### 二叉树后序遍历

> 思路1：递归
>
> 思路2：利用栈，同时建立一个`pre`指针，指向上一个访问的节点。用于判断当前节点的右子树是否已被访问。
>
> 思路3：`Morris`遍历，与`Morris`中序遍历类似。
>
> 个人理解：这个Morris后序遍历。实质上还是采用Morris中序遍历的思路。但是由于中序遍历的顺序为左->根->右。后序遍历为左->右->根。第一次添加节点是相当于添加一次左。反转一次还是左。然后往res里添加根,右。反转后变为 左->右->根。左子树已经添加完毕。后面的话仅需添加根节点。右子树即可，每次添加相当于也就是反转为右->根，连起来也就是左->右->根。

```c++
class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) {
        TreeNode *pre = nullptr;  // 用于记录上一个访问的节点
        TreeNode *cur = root;     // 当前处理的节点
        stack<TreeNode *> st;     // 栈用于辅助遍历
        vector<int> res;          // 存储遍历结果
        
        while (cur || !st.empty()) {
            // 一直向左走，先处理左子树
            while (cur) {
                st.push(cur);     // 将当前节点入栈
                cur = cur->left;  // 移动到左子节点
            }
            
            // 左子树处理完毕，处理栈顶元素
            cur = st.top();
            
            // 判断当前节点的右子树是否已被访问或右子树为空
            if (cur->right == nullptr || cur->right == pre) {
                res.push_back(cur->val); // 访问当前节点
                pre = cur;               // 设置pre为当前节点
                st.pop();                // 弹出栈顶元素
                cur = nullptr;           // 当前节点置空，准备处理下一个节点
            } else {
                // 如果右子树未被访问，移动到右子节点
                cur = cur->right;
            }
        }
        
        return res; // 返回后序遍历结果
    }
};

//Morris遍历
class Solution {
public:
    void addPath(vector<int> &vec, TreeNode *node) {
        int count = 0;
        while (node != nullptr) {
            ++count;
            vec.emplace_back(node->val);
            node = node->right;
        }
        reverse(vec.end() - count, vec.end()); // 将添加的这段路径进行反转
    }

    vector<int> postorderTraversal(TreeNode *root) {
        vector<int> res;
        if (root == nullptr) {
            return res;
        }
        TreeNode *p1 = root, *p2 = nullptr;

        while (p1 != nullptr) {
            p2 = p1->left;
            if (p2 != nullptr) {
                // 找到前驱节点
                while (p2->right != nullptr && p2->right != p1) {
                    p2 = p2->right;
                }
                if (p2->right == nullptr) {
                    // 建立临时线索指针
                    p2->right = p1;
                    p1 = p1->left;
                    continue;
                } else {
                    // 恢复树的结构
                    p2->right = nullptr;
                    // 添加路径并反转
                    addPath(res, p1->left);
                }
            }
            p1 = p1->right;
        }
        // 最后处理根节点到当前节点的路径
        addPath(res, root);
        return res;
    }
};

```

#### Morris先序/中序总结

> 大致过程相同，不同之处仅在于访问节点时机不同，访问节点位置不同
>
> 1. **访问节点的时机不同**：
>    * **前序遍历**：先访问根节点，然后遍历左子树，最后遍历右子树。在找到前驱节点并建立临时线索时就访问当前节点。
>    * **中序遍历**：先遍历左子树，访问根节点，然后遍历右子树。只有在左子树遍历完成，断开临时线索时才访问当前节点。
> 2. **访问节点的位置不同**：
>    - **前序遍历**：在判断前驱节点的右指针为空并建立临时线索时访问当前节点。
>    - **中序遍历**：在恢复树的结构，断开临时线索后访问当前节点。

#### 层序遍历（dfs实现待做）

> 思路1：用队列进行遍历，同时采用两个变量记录当前层节点数量，以及下一层节点数量。`oldcount`,`newcount`.

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
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> res;  // 存储最终结果的二维数组
        queue<TreeNode *> qu;     // 用于层序遍历的队列
        TreeNode *p = root;       // 当前访问的节点
        int oldcount = 0;         // 记录当前层的节点数量
        int newcount = 0;         // 记录下一层的节点数量
        
        if (p) {                  // 如果根节点不为空
            qu.push(p);           // 将根节点加入队列
            oldcount = 1;         // 当前层的节点数量设为1
        }
        while (!qu.empty()) {     // 当队列不为空时
            vector<int> flood;    // 用于存储当前层的节点值
            
            while (oldcount--) {  // 遍历当前层的所有节点
                p = qu.front();   // 获取队列的头节点
                qu.pop();         // 弹出队列的头节点
                flood.push_back(p->val);  // 将节点值加入当前层的结果中
                
                if (p->left) {     // 如果左子节点不为空
                    qu.push(p->left);  // 将左子节点加入队列
                    newcount++;        // 下一层的节点数量加1
                }
                if (p->right) {    // 如果右子节点不为空
                    qu.push(p->right); // 将右子节点加入队列
                    newcount++;        // 下一层的节点数量加1
                }
            }
            
            // 更新当前层节点数量，同时将当前层结果加入最终结果中
            oldcount = newcount;
            newcount = 0;
            res.push_back(flood);
        }
        
        return res;  // 返回最终的层序遍历结果
    }
};

```

### 根据序列构造二叉树



#### 前序中序构造二叉树

> 思路：具体思路为，前序遍历为`根->左->右`。中序遍历为`左->根->右`。每次找到前序遍历数组中第一个值在中序遍历数组中的位置，将其分割成两部分，左边一部分即为以当前节点为根的左子树，计算左子树长度`left_size`。用左子树长度，将前序和中序遍历中左子树集合与右子树集合，分离出来。又变为了子问题。用递归进行实现。
>
> ### 关键点
>
> 1. `vector<int> pre1(preorder.begin()+1,preorder.begin()+1+left_size);`构造区间为**左开右闭区间**，不包括最后一个元素。
> 2. 中序遍历集合要从第一个元素开始截取，跳过当前元素
> 3. 前序遍历集合当前元素已经确定，根据left_size 构造左子树和右子树，从第二个元素开始截取。左子树集合为`[1,1+left_size)`.右子树集合为`[1+left_size,end())`

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
class Solution {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        if(preorder.empty())
            return nullptr;

        int left_size=find(inorder.begin(),inorder.end(),preorder[0])-inorder.begin();
        //构造左子树集合
        vector<int> pre1(preorder.begin()+1,preorder.begin()+1+left_size);
        //构造右子树前序集合
        vector<int> pre2(preorder.begin()+1+left_size,preorder.end());

        //构造左子树中序集合
        vector<int> in1(inorder.begin(),inorder.begin()+left_size);
        //构造右子树中序集合
        vector<int> in2(inorder.begin()+1+left_size,inorder.end());
        TreeNode *left=buildTree(pre1,in1);
        TreeNode *right=buildTree(pre2,in2);
        //返回创建好的子树
        return new TreeNode(preorder[0],left,right);
    }
};
```

#### 中序后序构造二叉树

> 思路：中序遍历`左->根->右`，后序遍历`左->右->根`。跟上题类似，只不过根节点为后序遍历的末尾，找到根节点在中序遍历的位置，分割成左右两颗子树。计算左子树的长度`left_size`.构造中序和后序遍历的左右子树集合。
>
> ### 关键点
>
> 1. 后序遍历的根节点是末尾元素
> 2. 后序构造左子树是从开头开始，前序是从第二个元素开始。（不一致，注意看访问顺序`左->右->根`）

```c++
```

#### 前序后序构造二叉树

### BST题目

#### [98. 验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/)

> ### 二叉搜索树（BST）
>
> 节点的值：左<根<右
>
> 思路：总共有三种判别方式，分别都是依据先序遍历中序遍历和后序遍历。
>
> 1.  先序遍历（**节点的范围往下传递**）：对于每个子树，维护一个节点取值的范围区间`[left,right]`，`left`表示该点能存放的最小值，`right`表示该点能够存放的最大值。必须满足`left<x<right`。如果递归左子树就更新右区间`[left,root->val]`，如果更新右子树就更新左区间`[root->val,right]`
>
> 2. 中序遍历：由于二叉搜索树满足条件`左<根<右`,这恰好是中序遍历的顺序，因此我们仅需比较在中序序列中，当前节点与前一个访问节点值的大小即可，如果满足`root->val<=pre`说明不满足。继续递归右子树。
>
> 3. :imp: 后序遍历（**节点范围往上传**）：对于每个子树的状态，我们返回一个状态范围，`[l_min,r_max]`。这代表什么意思呢？**表示以当前节点t为根的子树中的最小值和最大值**，也就是当前子树所有的节点所能取到的范围。往上进行递归返回。接收到左右子树的最小值和最大值（分别的）。`[L_min,L_max],[R_min,R_max]`。与当前根节点t进行一个比较，判断是否满足BST的定义
>
>    `L_max<t<R_min`如果发现不满足。返回状态范围`[-INF,INF]`。为什么这样返回呢？这样返回可以保证该子树向上返回给上一层根节点时候。能够让`L_max<t<R_min`判断出错。最后进行判断若最后返回的`pair`对中第二个`second`的值为INF代表不是`BST`
>
>    * 关键点：
>      * 空节点一定是BST：返回`[INF,-INF](min,max)`一定满足条件。
>      * 如果当前子树不是BST：返回`[-INF,INF](min,max)`一定不满足条件。
>      * 检查是否满足：`check.second!=INF`

```c++
class Solution {
public:
    //中序遍历
     long pre=-LONG_MAX;
    bool check(TreeNode *root){
        if(root==nullptr)
            return true;
        bool flagL=check(root->left);
        if(!flagL||root->val<=pre)return false;
        pre=root->val;

        return check(root->right);
    }
    bool isValidBST(TreeNode* root) {
        return check(root);
    }
};


//后序遍历
class Solution {
public:
    // 定义一个长整型的pair类型，PLL，用于存储子树的最小值和最大值
    typedef pair<long, long> PLL;
    // 定义一个常量INF，用于表示极端的值
    static const long INF = LONG_MAX;

    // 辅助函数check，用于递归检查子树是否为有效的BST
    PLL check(TreeNode* root) {
        // 如果当前节点为空，返回极端值，表示这个子树是一个有效的空BST
        if (root == nullptr)
            return {INF, -INF};

        // 递归检查左子树和右子树
        auto [L_min, L_max] = check(root->left);
        auto [R_min, R_max] = check(root->right);

        // 获取当前节点的值
        long x = root->val;

        // 检查当前节点是否满足BST的条件：左子树的最大值小于当前节点值，且右子树的最小值大于当前节点值
        if (L_max >= x || x >= R_min)
            return {-INF, INF}; // 如果不满足条件，返回极端值，表示这个子树不是有效的BST

        // 返回当前子树的最小值和最大值，分别为左子树的最小值和右子树的最大值
        return {min(L_min, x), max(R_max, x)};
    }

    // 主函数isValidBST，用于判断整个树是否为有效的BST
    bool isValidBST(TreeNode* root) {
        // 调用辅助函数check，如果返回的最大值不是极端值INF，表示这是一个有效的BST
        return check(root).second != INF;
    }
};

//前序遍历
class Solution {
public:
    // 定义一个常量 INF，用于表示极限值（正无穷）
    static const long INF = LONG_MAX;

    // 辅助函数 check，用于递归检查每个子树是否为有效的二叉搜索树（BST）
    // 参数：root - 当前节点，left - 当前节点值应该大于的左界限，right - 当前节点值应该小于的右界限
    bool check(TreeNode* root, long left, long right) {
        // 如果当前节点为空，返回 true，表示这是一个有效的空BST
        if (root == nullptr)
            return true;
        
        // 检查当前节点的值是否在允许的区间内
        if (root->val <= left || root->val >= right)
            return false;
        
        // 递归检查左子树和右子树，更新区间范围
        return check(root->left, left, root->val) && check(root->right, root->val, right);
    }

    // 主函数 isValidBST，用于判断整个树是否为有效的BST
    bool isValidBST(TreeNode* root) {
        // 调用辅助函数 check，初始区间为负无穷到正无穷
        return check(root, -INF, INF);
    }
};

```

#### [2476. 二叉搜索树最近节点查询](https://leetcode.cn/problems/closest-nodes-queries-in-a-binary-search-tree/)

> 思路：利用BST中序遍历是一个递增的序列，因此采用二分查找进行搜索`>=x`的第一个元素。

```c ++
class Solution {
public:
    // 中序遍历函数，将节点值按顺序存入 nums 数组
    void inorderTraversal(TreeNode* root, vector<int>& nums) {
        if (!root) return; // 如果当前节点为空，直接返回
        inorderTraversal(root->left, nums); // 递归遍历左子树
        nums.push_back(root->val); // 将当前节点的值存入 nums 数组
        inorderTraversal(root->right, nums); // 递归遍历右子树
    }

    // 二分查找函数，找到第一个大于等于 x 的位置
    int lower_bound(const vector<int>& nums, int x) {
        int left = 0, right = nums.size(); // 初始化左指针和右指针
        while (left < right) { // 当左指针小于右指针时，继续查找
            int mid = left + (right - left) / 2; // 计算中间位置
            if (nums[mid] < x) left = mid + 1; // 如果中间值小于 x，移动左指针
            else right = mid; // 否则，移动右指针
        }
        return left; // 返回第一个大于等于 x 的位置
    }

    vector<vector<int>> closestNodes(TreeNode* root, vector<int>& queries) {
        vector<int> nums; // 存储中序遍历结果的数组
        vector<vector<int>> res; // 存储结果的二维数组

        // 获取中序遍历的有序数组
        inorderTraversal(root, nums);

        // 遍历每个查询
        for(auto query:queries){
            int pos = lower_bound(nums, query); // 查找第一个大于等于 query 的位置
            vector<int> path(2, -1); // 初始化路径数组，默认值为 -1

            // 找到小于等于 query 的最大值
            if (pos >= 0) {
                if (pos < nums.size() && nums[pos] == query)
                    path[0] = nums[pos]; // 如果找到的值等于 query，直接存入 path[0]
                else if (pos > 0)
                    path[0] = nums[pos - 1]; // 否则存入前一个位置的值
            }

            // 找到大于等于 query 的最小值
            if (pos < nums.size())
                path[1] = nums[pos]; // 存入找到的值

            res.emplace_back(path); // 将路径数组存入结果数组
        }
        return res; // 返回结果数组
    }
};
```

#### [1373. 二叉搜索子树的最大键值和](https://leetcode.cn/problems/maximum-sum-bst-in-binary-tree/)

> 思路：利用验证二叉搜索树的后序遍历检测，使用一个元组`tuple`记录3个值,`[min,max,sum]`,分别代表以当前节点为根节点的**子树的最小值，最大值，所有节点和**。如果检测当前子树不是BST那么返回`[-INF,INF，0]`即可代表当前子树不是`BST`。采用一个`res`变量，每次跟s取最大值。
>
> ### 关键点
>
> 1. 多增加一个s的值，采用元组存储三个值。
> 2. 检测失败返回`[-INF,INF，0]`

```c++
class Solution {
public:
    // 定义一个静态常量，用于表示正无穷大
    static const int INF = INT_MAX;
    
    // 记录最大 BST 和的成员变量
    int res = 0;

    // 辅助函数，返回三个值: 
    // 1. 当前子树的最小值
    // 2. 当前子树的最大值
    // 3. 当前子树的和
    tuple<int, int, int> find(TreeNode *root) {
        if (root == nullptr) {
            // 空树是一颗 BST，返回最大值、最小值和和都为零
            return {INF, -INF, 0};
        }

        // 递归处理左子树，获取左子树的最小值、最大值和和
        auto [L_MIN, L_MAX, L_SUM] = find(root->left);
        // 递归处理右子树，获取右子树的最小值、最大值和和
        auto [R_MIN, R_MAX, R_SUM] = find(root->right);

        // 检查当前树是否为 BST:
        // 当前节点值必须大于左子树的最大值，并且小于右子树的最小值
        if (root->val <= L_MAX || root->val >= R_MIN) {
            // 如果不是 BST，返回无效的值
            return {-INF, INF, 0};
        }

        // 如果当前树是一个 BST，计算当前树的和
        int s = L_SUM + R_SUM + root->val;
        // 更新最大 BST 和
        res = max(res, s);

        // 返回当前树的最小值、最大值和当前树的和
        return {min(L_MIN, root->val), max(R_MAX, root->val), s};
    }

    // 主函数，调用 find 函数并返回结果
    int maxSumBST(TreeNode* root) {
        find(root);
        return res;
    }
};

```

### 公共祖先

#### [236. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/)

> 思路：分类讨论，如果当前节点为`null`,`p`或者`q`。直接返回当前节点即可，得到左右子树递归结果，如果左右子树递归结果均不为空，说明，当前root节点即为最近的祖先节点。返回当前节点`root`（原因是，`p,q`分别在左右不同的子树之中）。有一个不为空，返回不为空的结果即可。

```c++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        // 如果当前节点为空，或者当前节点是p或q中的一个，返回当前节点
        if (root == NULL || root->val == p->val || root->val == q->val)
            return root;
        
        // 递归查找左子树中的LCA
        TreeNode *left = lowestCommonAncestor(root->left, p, q);
        // 递归查找右子树中的LCA
        TreeNode *right = lowestCommonAncestor(root->right, p, q);
        
        // 如果左子树和右子树分别找到了p和q，那么当前节点root就是LCA
        if (left && right)
            return root;
        
        // 否则，返回非空的递归结果，意思就是找到的p，或者q
        return left ? left : right;
    }
};

```

#### [235. 二叉搜索树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-search-tree/)

> 思路：分类讨论，根据`BST`的特性，满足`左<根<右`。主要以下情况，
>
> 1. 当前节点为`null，p，q`。直接返回当前节点即可。
> 2. 如果，`p,q`仅在当前节点的左侧或者右侧，返回在左侧或者右侧的递归结果即可。
> 3. 如果在两侧，返回当前节点即可

```c++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        // 如果当前节点为空，直接返回空节点
        if (root == NULL)
            return root;

        // 获取当前节点的值
        int x = root->val;

        // 如果p和q的值都小于当前节点的值，说明它们都在左子树中
        if (p->val < x && q->val < x)
            return lowestCommonAncestor(root->left, p, q);

        // 如果p和q的值都大于当前节点的值，说明它们都在右子树中
        if (p->val > x && q->val > x)
            return lowestCommonAncestor(root->right, p, q);

        // 否则，说明p和q分布在当前节点的两侧（一个在左，一个在右），当前节点就是LCA
        return root;
    }
};

```

#### [1123. 最深叶节点的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-deepest-leaves/)

> 思路：第一种思路为从上往下进行遍历，维护一个最大深度，如果当前节点的左右子树所在的深度等于最大深度，更新res。（最大深度不一定在第一时间就更新完成，因此当前更新只是暂时的进行更新）返回当前子树的最大深度。
>
> **思路2**：从下往上进行归，将每颗子树看成是一个子问题，那么，需要解决的问题
>
> 1. 子树视角下最深叶节点的深度，就是子树的高度。
> 2. 子树的最深叶节点的最近公共祖先`LCA`
>
> 分为以下几种情况：
>
> 1. 如果`lh<rh`，当前子树的高度为`rh+1,`最近公共祖先节点为右子树的`LCA`
> 2. 如果`rh<lh`，当前子树的高度为`lh+1`，最近公共祖先节点为左子树的`LCA`
> 3. 如果`lh==rh`，当前子树的高度为`lh+1`，最近公共祖先节点为`node`
>
> ### 关键点
>
> 1. 返回一个`pair`类型，分别存放当前子树的高度，当前子树的LCA；
> 2. 三种情况的讨论

```c++
//法一：维护最大深度，由上往下传递
class Solution {
public:
    int max_deep=-1;
    TreeNode *res=nullptr;
    int deepth(TreeNode *root,int height){
        if(root==nullptr){
            max_deep=max(max_deep,height);//维护一个全局最大深度
            return height;
        }
        int lh=deepth(root->left,height+1);//左子树最大深度
        int rh=deepth(root->right,height+1);//右子树最大深度
        //获取到左右子树深度后，判断是否等于最大深度
        if(lh==max_deep&&rh==max_deep)
            res=root;
        //返回当前子树最深叶节点的深度。
        return max(lh,rh);
    }
    TreeNode* lcaDeepestLeaves(TreeNode* root) {
        deepth(root,0);
        return res;
    }
};

//法二：从下往上进行归，返回子树的最大深度以及临时祖先节点。
class Solution {
public:
    typedef pair<int,TreeNode *>PIT;
    PIT findLCA(TreeNode *root){
        if(root==nullptr)
            return {0,nullptr};
        auto [Lh,LCAL]=findLCA(root->left);
        auto [Rh,LCAR]=findLCA(root->right);
        if(Lh<Rh)//右子树更高，一定在右子树
        return {Rh+1,LCAR};
        if(Lh>Rh)//左子树更高，一定在左子树
        return {Lh+1,LCAL};
        else //一样高，当前节点即为LCA
        return {Lh+1,root};
    }
    TreeNode* lcaDeepestLeaves(TreeNode* root) {
        return findLCA(root).second;
    }
};
```

### BFS（使用可以记录层次的模板）

#### [102. 二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/)

> 思路：套用图的BFS能够记录当前层的模板，二叉树在循环内部仅访问左右孩子节点即可，图为访问邻接表。

```c++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> res;
        queue<TreeNode *> q;

        if(root==nullptr)
            return res;
        q.push(root);
        while(q.size()){
            vector<int> path;
            int size=q.size();//记录这一层有多少个元素
            for(int i=0;i<size;i++){
                TreeNode *node=q.front();
                q.pop();
                path.emplace_back(node->val);
                //如果是图，此处为遍历邻接表
                if(node->left)q.push(node->left);
                if(node->right)q.push(node->right);
            }
            res.emplace_back(path);
        }
        return res;
    }
};
```

#### [103. 二叉树的锯齿形层序遍历](https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/)

> 思路：题目偶数层是需要进行一个逆置，因此我们要使用一个`level`变量进行计算当前层到底是偶数层还是奇数层。其他与上题一致。

```c++
class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>> res; // 用于存储最终的锯齿形层序遍历结果

        queue<TreeNode*> q; // 用于辅助层序遍历的队列
        if (root == nullptr) // 如果根节点为空，返回空结果
            return res;

        q.push(root); // 将根节点加入队列
        int level = 0; // 用于记录当前层数

        while (!q.empty()) { // 当队列不为空时，进行层序遍历
            level++; // 增加当前层数
            int size = q.size(); // 记录当前层的节点数
            vector<int> path; // 用于存储当前层的节点值

            for (int i = 0; i < size; ++i) { // 遍历当前层的所有节点
                TreeNode* temp = q.front(); // 获取当前层的第一个节点
                q.pop(); // 弹出该节点
                path.emplace_back(temp->val); // 将该节点的值加入当前层的结果

                // 如果左子节点存在，将其加入队列
                if (temp->left) q.push(temp->left);
                // 如果右子节点存在，将其加入队列
                if (temp->right) q.push(temp->right);
            }

            // 如果当前层是偶数层，需要将当前层的结果反转
            if (level % 2 == 0)
                reverse(path.begin(), path.end());

            // 将当前层的结果加入最终结果中
            res.emplace_back(path);
        }

        return res; // 返回最终的锯齿形层序遍历结果
    }
};

```

#### [513. 找树左下角的值](https://leetcode.cn/problems/find-bottom-left-tree-value/)

> 思路：经典的`BFS`遍历，用`res`记录返回值，访问每一层节点的时候，判断一下当前是否是第一个节点。如果是，更新`res`即可。
>
> 思路2：将加入孩子节点的时候，变为先加入右孩子节点，这样就变成了`根->右->左`。最后更新的`res`即为最左边第一个。

```c++
class Solution {
public:
    int findBottomLeftValue(TreeNode* root) {
        int res=-1;
        queue<TreeNode *> q;
        q.push(root);

        while(q.size()){
            int size=q.size();
            for(int i=0;i<size;i++){
                auto temp=q.front();
                q.pop();
                res=temp->val;
                if(temp->right)q.push(temp->right);
                if(temp->left)q.push(temp->left);
            }
        }
        return res;
    }
};
```

#### [2583. 二叉树中的第 K 大层和](https://leetcode.cn/problems/kth-largest-sum-in-a-binary-tree/)

> 思路：采用一个数组或者大根堆来进行存储每一层的和。通过BFS进行计算每一层的和。最后将大根堆的前k-1个出队即可，最后的top即为第K大和。

```c++
class Solution {
public:
typedef long long LL;
    static bool cmp(LL a,LL b){
        return a>b;
    }
    
    long long kthLargestLevelSum(TreeNode* root, int k) {
        priority_queue<LL> ans;
        queue<TreeNode *> q;
        if(root==nullptr)
            return -1;
        q.push(root);
        while(q.size()){
            int size=q.size();
            LL sum=0;
            for(int i=0;i<size;i++){
                auto temp=q.front();
                q.pop();
                sum+=temp->val;
                if(temp->left)q.push(temp->left);
                if(temp->right)q.push(temp->right);
            }
            ans.push(sum);
        }
        if(ans.size()<k)return -1;
        for(int i=0;i<k-1;i++)
            ans.pop();
        return ans.top();
    }
};
```

#### [116. 填充每个节点的下一个右侧节点指针](https://leetcode.cn/problems/populating-next-right-pointers-in-each-node/)

> 思路：利用层序遍历，每一次遍历一层的时候，给这一层加上链表指针即可。由于需要访问前一个节点，因此，我们可以使用数组来进行模拟队列。

```c++
class Solution {
public:
    static const int N = 5050;  // 定义一个足够大的数组来模拟队列
    Node* q[N];  // 队列数组
    int tt = -1; // 队尾指针
    int hh = 0;  // 队头指针
    
    Node* connect(Node* root) {
        if (root == NULL)
            return root;
        
        q[++tt] = root;  // 将根节点加入队列

        while (hh <= tt) {
            int size = tt - hh + 1;  // 当前层的节点数
            
            for (int i = 0; i < size; i++) {
                Node* temp = q[hh + i];  // 获取当前层的节点

                // 将当前层节点相连
                if (i > 0)
                    q[hh + i - 1]->next = temp;
                
                // 将左右子节点加入队列
                if (temp->left)
                    q[++tt] = temp->left;
                if (temp->right)
                    q[++tt] = temp->right;
            }

            hh += size;  // 更新队头指针，跳过当前层的节点
        }

        return root;
    }
};

```

#### [2415. 反转二叉树的奇数层](https://leetcode.cn/problems/reverse-odd-levels-of-binary-tree/)

> 思路：利用层序遍历，记录当前所在的层次，如果为奇数，则进行反转值即可。
>
> 思路2：利用DFS做法，将左孩子左节点与右孩子右节点值进行交换，同时左孩子右节点与右孩子左节点进行交换。

```c++
class Solution {
public:
    void dfs(TreeNode *p,TreeNode* q,int level){
        if(p==nullptr)
            return ;
        if(level%2==1){
        swap(p->val,q->val);
        }
        dfs(p->left,q->right,level+1);
        dfs(p->right,q->left,level+1);
    }
    TreeNode* reverseOddLevels(TreeNode* root) {
        dfs(root->left,root->right,1);
        return root;
    }
};

class Solution {
public:
    static const int N=2e4;
    TreeNode *q[N];
    int tt=-1,hh=0;
    TreeNode* reverseOddLevels(TreeNode* root) {
        q[++tt]=root;
        int level=-1;
        while(hh<=tt){
            level++;
            int size=tt-hh+1;
            vector<TreeNode *> lis;
            for(int i=0;i<size;i++){
                auto temp=q[hh++];
                if(level%2==1)
                lis.emplace_back(temp);
                if(temp->left)q[++tt]=temp->left;
                if(temp->right)q[++tt]=temp->right;
            }
            //反转
            if(level%2==1)
            for(int i=0,j=size-1;i<j;i++,j--)
            swap(lis[i]->val,lis[j]->val);
        }
        return root;
    }
};
```

#### [2641. 二叉树的堂兄弟节点 II](https://leetcode.cn/problems/cousins-in-binary-tree-ii/)

> 思路：利用BFS思路，计算每一层的和，由于我们要使用当前层节点去更新下一层的节点，因此要采用自定义的队列，或者利`用vector`模拟。计算完下一层的和之后，利用当前层的节点去更新下一层即可。也就是两次BFS

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr, right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    // 替换树中每个节点的值为其同层节点的总和减去其子节点值之和
    TreeNode* replaceValueInTree(TreeNode* root) {
        if (root == nullptr)
            return nullptr;

        vector<TreeNode*> q;  // 用于层序遍历的队列
        q.emplace_back(root);

        // 层序遍历，计算每层节点值的总和
        while (!q.empty()) {
            int size = q.size();  // 当前层的节点数量
            vector<TreeNode *> nxt;  // 存储下一层的节点
            int sum_t = 0;  // 下一层的节点值总和
            for (int i = 0; i < size; ++i) {
                auto temp = q[i];
                if (temp->left) {
                    sum_t += temp->left->val;  // 累加左子节点的值
                    nxt.emplace_back(temp->left);  // 将左子节点加入下一层队列
                }
                if (temp->right) {
                    sum_t += temp->right->val;  // 累加右子节点的值
                    nxt.emplace_back(temp->right);  // 将右子节点加入下一层队列
                }
            }
            // 更新当前层所有节点的子节点的值
            for (auto node : q) {
                int childSum = (node->left ? node->left->val : 0) + (node->right ? node->right->val : 0);  // 当前节点子节点的值之和
                int sub = sum_t - childSum;  // 新值为下一层节点总和减去当前节点子节点的值之和
                if (node->left)
                    node->left->val = sub;  // 更新左子节点的值
                if (node->right)
                    node->right->val = sub;  // 更新右子节点的值
            }
            q.assign(nxt.begin(), nxt.end());  // 更新当前队列为下一层的队列
        }
        root->val = 0;  // 根节点值设置为0
        return root;
    }
};

```

## 5.二分法:imp:

> `lower_bound(vector<int> &nums,int target)`：寻找数组中第一个**大于等于**`target`的位置。
>
> `upper_bound(vector<int> &nums,int target)`：寻找数组中第一个**大于**`target`的位置。

> 理解：想要理解二分法，就要明白二分法的精髓在于**循环不变量**也就是不管循环前后。代表的意义均不发生改变。大致可以将二分法分为三种。同时需要保证`[0-left],[left,right],[right,nums.size()-1]`前后两个区间的开闭，是由中间`[left,right]`这个区间决定的。同时这三个区间需要涵盖整个数组。决定了开闭之后。即定义了循环不变量。需要时刻保证循环不变量的正确性。
>
> 1. 左闭右闭区间`[left,right]`：
>
>    应用此区间。那么对于左区间即为`[0,left)->等价为[0,left-1]`,右区间为`(right,nums.size()-1]`。此时即可定义循环 不变量为：`left`左边一定小于`target`.`right`右边一定大于等于`target`。
>
>    * `nums[left-1]<target`
>    * `nums[right+1]>=target`
>
>    当出现`nums[mid]==target`时。按照循环不变量的定义此时应该更新`right=mid-1`。才满足`right`的右边一定大于等于`target`
>
>    最后当循环条件`while(left<=right)`不满足后。也就是当`left=right+1`后退出循环。此时`left`指向第一个大于等于`target`元素。
>
> 2. 左闭右开区间`[left,right)`：
>
>    此区间。对于左区间为`[0,left)`，右区间为`[right,nums.size())`.此时循环不变量为。`left`左边一定小于`target`.`right`以及`right`右边一定大于等于`target`.
>
>    * `nums[left-1]<target`
>    * `nums[right]>=target`
>
>    当出现`nums[mid]==target`时，此时按照循环不变量定义。应该更新`right=mid`.才满足`right`以及`right`右边一定大于等于`target`
>
>    最后当循环条件`while(left<right)`不满足时。也就是当`left=right`时退出循环。此时`left`指向第一个大于等于`target`元素。
>
> 3. 左开右开区间`(left,right)`：
>
>    此区间。对于左区间为`[0,left]`，右区间为`[right,nums.size())`.此时循环不变量为。`left`以及`left`左边一定小于`target`.`right`以及`right`右边一定大于等于`target`.
>
>    * `nums[left]<target`
>    * `nums[right]>=target`
>
>    当出现`nums[mid]==target`时，此时按照循环不变量定义。应该更新`right=mid`.才满足`right`以及`right`右边一定大于等于`target`
>
>    最后当循环条件`while(left+1<right)`不满足时。也就是当`left+1=right`时退出循环。此时`right`指向第一个大于等于`target`元素。
>
>    :smiling_imp:注：二分法的最后答案结果，不一定在区间内，有可能在区间外。
>
> ------------
>
> :apple:四种情况转换（`>,>=,<,<=`）
>
> **查找第一个大于目标值的位置 (`>`)：**
>
> - 使用 `upper_bound`。同样也可以使用`lower_bound(val+1)`
>
> **查找第一个大于或等于目标值的位置 (`>=`)：**
>
> - 使用 `lower_bound`。或者`upper_bound(val-1)`
>
> **查找第一个小于目标值的位置 (`<`)：**
>
> - 使用 `lower_bound`，然后减一。或者`lower_bound(val-1)`
>
> **查找第一个小于或等于目标值的位置 (`<=`)：**
>
> - 使用 `upper_bound`，然后减一。或者`upper_bound(val - 1)`

```c++
//左闭右闭区间 [left, right]

/**nums[left-1] < target
nums[right+1] >= target**/
int binarySearch(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return left;  // 此时 left 指向第一个大于等于 target 的元素或者right+1
}
/**
左闭右开区间 [left, right)

nums[left-1] < target
nums[right] >= target
**/
int binarySearch(vector<int>& nums, int target) {
    int left = 0, right = nums.size();
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target)
            left = mid + 1;
        else
            right = mid;
    }
    return left;  // 此时 left 指向第一个大于等于 target 的元素或者right
}
/**
左开右开区间 (left, right)

nums[left] < target
nums[right] >= target
**/
int binarySearch(vector<int>& nums, int target) {
    int left = -1, right = nums.size();
    while (left + 1 < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target)
            left = mid;
        else
            right = mid;
    }
    return right;  // 此时 right 指向第一个大于等于 target 的元素,或者left+1
}

```



### 剑指172统计目标成绩出现次数

> 思路：用两次二分查找分别找出`target`左边界和`target`右边界.此方法左边界和右边界都是第一个不等于`target`的值
>
> 解法：
>
> 1. 初始化： 左边界 `i=0` ，右边界 `j=len(scores)−1` 。
> 2. 循环二分： 当闭区间`[i, j]` 无元素时跳出；
>    * 计算中点 `m=(i+j)/2` （向下取整）；
>    * 若` scores[m]<target`，则 `target` 在闭区间`[m+1,j] `中，因此执行` i=m+1`；
>    * 若 `scores[m]>target`，则 `target` 在闭区间`[i,m−1]` 中，因此执行` j=m−1`；
>    * 若 `scores[m]=target` ，则右边界 `right`在闭区间`[m+1,j]` 中；左边界 `left` 在闭区间`[i,m−1]` 中。因此分为以下两种情况：
>      * 若查找 右边界` right `，则执行 `i=m+1` ；（跳出时 `i` 指向右边界）
>      * 若查找 左边界 `tleft` ，则执行` j=m−1`；（跳出时 `j` 指向左边界）

```c++
//由于是有序数组，因此我们只需要查找target右边界，以及target-1的右边界。二者相减即可

class Solution {
public:
    int countTarget(vector<int>& scores, int target) {
        return loc(scores,target)-loc(scores,target-1);
    }
    //查找右边界
    int loc(vector<int> &scores,int tar){
        int left=0,right=scores.size()-1;
        while(left<=right){
            int mid=left+(right-left)/2;
            if(scores[mid]<=tar)
                left=mid+1;
            else
                right=mid-1;
        }
        return left;
    }
};
```

### [搜索插入的位置](https://leetcode.cn/problems/search-insert-position/)

> 本题解法可以由两个思路。分别是将区间看做是`[left,right]`的左闭右闭区间。以及`[left,right)`的左闭右开区间。
>
> 这两种写法有不同，
>
> 1. 由于是左闭右闭区间，那么`left==right`是可以成立的。因此循环条件要写作`while(left<=right)`。同时当`nums[mid]>target`时。由于`nums[right]`是合法的值。因此可以取到。要更新`right=mid-1`.
> 2. 对于左闭右开区间。那么只可能有`left<right`这种情况。二者相等这种情况，是不能够取到的。因此循环条件只能够写作`while(left<right)`.同时当`nums[mid]>target`时。由于`nums[right]`是不能够取到的值。因此更新区间为`right=mid.  [left，mid)`

```c++
//左闭右闭区间
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int left=0;
        int right=nums.size()-1;
        int mid=0;
        while(left<=right){//不同之处
            mid=(right+left)/2;
            if(nums[mid]<target)
                left=mid+1;  
            else
                right=mid-1;//不同之处
        }
        return left;
    }
};


//左闭右开区间
class Solution {
public:
//采用左闭右开区间
    int searchInsert(vector<int>& nums, int target) {
        int left=0,right=nums.size();
        int mid=0;
        while(left<right){//不同之处
            mid=left+(right-left)/2;
            if(nums[mid]<target)
                left=mid+1;
            else 
                right=mid;//不同之处
        }
        return left;
    }
};
```

### 正整数和负整数的最大计数

> 思路：查找的第一个大于-1的数位置，查找到第一个大于0的位置。

```c++
class Solution {
public:
    int lower_bound(vector<int> &nums,int tar){
        //定义为左闭右闭区间
        int left=0,right=nums.size()-1;
        //
        int mid;
        while(left<=right){
            mid=left+(right-left)/2;
            if(nums[mid]<=tar)
                left=mid+1;
            else
                right=mid-1;
        }
        return left;
    }
    int maximumCount(vector<int>& nums) {
			//查找到第一个大于-1的位置。
        int left=lower_bound(nums,-1);
                //查找到第一个>0的位置。
        int right=lower_bound(nums,0);
        int fushu=left;
        int zhengshu=nums.size()-right;
        return max(fushu,zhengshu);
    }
};
```

### 咒语和药水的成功对数

> 思路：将`potions`数组升序排列。遍历`spells`数组。其中$$y \geq \left\lceil \frac{\text{success}}{x} \right\rceil$$即可。不等式变形为$$y > \left\lfloor \frac{\text{success} - 1}{x} \right\rfloor$$，考虑使用二分查找。即查找第一个大于`(success-1)/x`的数。

```c++
class Solution {
public:
    // 辅助函数，用于执行二分查找，找到第一个大于 tar 的位置
    int lower_pound(vector<int> &nums, long long tar) {
        int left = 0, right = nums.size();
        int mid = 0;
        while (left < right) {
            mid = left + (right - left) / 2;
            // 如果中间元素小于等于 tar，则搜索右半部分
            if (nums[mid] <= tar)
                left = mid + 1;
            else // 否则，搜索左半部分
                right = mid;
        }
        return left; // 返回第一个大于 tar 的元素的索引
    }

    // 函数用于找到每个 spells 中的元素与 potions 组合成功的次数
    vector<int> successfulPairs(vector<int>& spells, vector<int>& potions, long long success) {
        vector<int> res; // 结果向量
        sort(potions.begin(), potions.end()); // 对 potions 进行排序
        for (int i = 0; i < spells.size(); i++) {
            // 计算成功所需的最小值
            long long tar = (success - 1) / spells[i];
            // 使用 lower_pound 查找第一个大于 tar 的位置
            int left = lower_pound(potions, tar);
            if (left >= potions.size())
                res.push_back(0); // 如果没有找到，则返回0
            else
                // 返回 potions 中所有可能的成功组合数
                res.push_back(potions.size() - left);
        }
        return res; // 返回最终结果
    }
};

```

### 寻找比目标字母大的最小字母

> 思路：区间：左闭右开
>
> 循环不变量：
>
> `nums[left-1] <= target`（左边的元素小于等于目标值）
>
> `nums[right] > target`（右边的元素大于目标值）

```c++
class Solution {
public:
//寻找第一个大于tar的字符
    char lower_bound(vector<char> &nums,char target){
        //定义R以及右边大于target,L左边小于等于tar
        int left=0,right=nums.size();
        while(left<right){
            int mid=left+(right-left)/2;
            if(nums[mid]<=target)
                left=mid+1;
            else
                right=mid;
        }
        return left<nums.size()&&nums[left]>target?nums[left]:nums[0];
    }
    char nextGreatestLetter(vector<char>& letters, char target) {
        return lower_bound(letters,target);
    }
};

```

### 和有限的最长子序列

> 思路：本题，没有要求连续序列，因此可以对数组进行排序，从而使用二分。
>
> 解法：
>
> 1. 构造一个`vector<int> f`数组代表前`i`个数的前缀和。不包括`nums[i]`。
> 2. 遍历`queries`数组，找到`f`数组中第一个大于`queries[i]`的数字。不大于`queries[i]`的长度为`index-1`

```c++
#include <vector>
#include <algorithm>

class Solution {
public:
    // 自定义的 lower_bound 函数，用于查找第一个大于 tar 的数的位置，返回其下标
    int lower_bound(std::vector<int> &nums, int tar) {
        // 初始化左边界和右边界
        int left = 0, right = nums.size();
        int mid;

        // 循环不变量：
        // 在每次迭代结束后，nums[left-1] <= tar 且 nums[right] > tar
        while (left < right) {
            // 计算中间位置
            mid = left + (right - left) / 2;
            if (nums[mid] > tar)
                // 如果中间位置的数大于 tar，则收缩右边界
                right = mid;
            else
                // 否则收缩左边界
                left = mid + 1;
        }
        // 返回第一个大于 tar 的数的位置
        return left;
    }

    std::vector<int> answerQueries(std::vector<int>& nums, std::vector<int>& queries) {
        // 前缀和数组，f[i] 代表前 i 个数的和，初始化大小为 nums.size() + 1，全为 0
        std::vector<int> f(nums.size() + 1, 0);
        // 用于存储每个查询的结果
        std::vector<int> res(queries.size(), 0);

        // 对 nums 进行排序，以便计算前缀和
        std::sort(nums.begin(), nums.end());

        // 计算前缀和
        for (int i = 1; i <= nums.size(); i++) {
            f[i] = f[i - 1] + nums[i - 1];
        }

        // 处理每一个查询
        for (int i = 0; i < queries.size(); i++) {
            // 查找前缀和中第一个大于查询值 queries[i] 的位置
            int first_big = lower_bound(f, queries[i]);
            // 结果为该位置的前一个位置的下标
            res[i] = first_big - 1;
        }

        // 返回所有查询的结果
        return res;
    }
};

```

### 比较字符串最小字母出现频次

> 思路：运用额外空间`str`存储`f[words[i]]`，遍历`queries`数组，在`str`中寻找第一个大于`queries[i]`的位置。存储`len-index`到结果数组中。
>
> 解法：
>
> 1. 首先对 `words` 数组中的每个字符串进行排序，并计算每个字符串中最小字符的频率。然后将这些频率存储在 `str` 数组中，并对 `str` 数组进行排序。
>
> 2. 对于 `queries` 数组中的每个字符串，同样对其进行排序，并使用 `lower_bound` 函数找到 `str` 数组中大于其最小字符频率的位置，从而确定有多少个 `words` 的频率比其大。
>
> 3. 最后将结果存储在 `res` 数组中并返回。

```c++
class Solution {
public:
    // 查找第一个大于目标值 tar 的位置
    int lower_bound(vector<int>& nums, int tar) {
        // 定义左闭右闭区间
        int left = 0, right = nums.size() - 1;
        int mid;
        // 循环查找
        while (left <= right) {
            mid = left + (right - left) / 2; // 计算中间位置
            if (nums[mid] > tar)
                right = mid - 1; // 缩小右边界
            else
                left = mid + 1; // 缩小左边界
        }
        // 返回第一个大于目标值的位置
        return left < nums.size() ? left : nums.size();
    }

    // 假设字符串已经排好序，计算最小字符出现的频率
    int f1(string str, int len) {
        int count = 0; // 初始化计数器
        for (int i = 0; i < len; i++) {
            count++; // 计数
            // 如果遇到不同的字符，则返回当前计数
            if (i + 1 < len && str[i + 1] != str[i])
                return count;
        }
        return count; // 如果整个字符串都是相同字符，则返回总长度
    }

    vector<int> numSmallerByFrequency(vector<string>& queries, vector<string>& words) {
        // 定义一个 vector<int> 表示每个 word 的最小字符频率
        vector<int> str(words.size(), 0);
        // 定义一个 vector<int> 用于存储结果
        vector<int> res(queries.size(), 0);

        // 计算 words 中每个字符串的最小字符频率
        for (int i = 0; i < words.size(); i++) {
            sort(words[i].begin(), words[i].end()); // 排序字符串
            str[i] = f1(words[i], words[i].size()); // 计算最小字符频率
        }

        // 对频率数组进行排序
        sort(str.begin(), str.end());

        int len = words.size();
        // 计算 queries 中每个字符串的结果
        for (int i = 0; i < queries.size(); i++) {
            sort(queries[i].begin(), queries[i].end()); // 排序字符串
            // 使用 lower_bound 查找大于当前最小字符频率的位置
            int index = lower_bound(str, f1(queries[i], queries[i].size()));
            // 计算有多少个 words 的最小字符频率比当前 query 大
            res[i] = len - index;
        }

        return res; // 返回结果
    }
};

```

### 区间内查询数字的频率(第一次超时)

> 思路：使用`map<int,vector<int>>`，缓存数组中相同数字的位置。由于下标是排好序的，因此我们可以使用二分法。
>
> 例如：对于$$query(3,5,2)$$​，那么 2 的下标列表 [1,4,5] 中的下标 4 和 5 就是满足要求的，返回 2。
>
> 解法：把下标列表记作数组 *a*，由于 *a* 是**有序**数组，我们可以用**二分查找**快速求出：
>
> 1. `a` 中的第一个` ≥left` 的数的下标，设其为 `p`。如果不存在则 `p` 等于 `a` 的长度。
> 2. `a` 中的第一个 `>right` 的数的下标，设其为 `q`。如果不存在则` p` 等于 `a` 的长度。
> 3. `a `中的下标在 `[p,q)` 内的数都是满足要求的，这有 `q−p` 个。如果 `a` 中不存在这样的数，那么 `q−p=0`，也符合要求。

```c++
class RangeFreqQuery {
public:
    unordered_map<int,vector<int>> cache;
    RangeFreqQuery(vector<int>& arr) {
        //构造 num与下标的映射关系
        for(int i=0;i<arr.size();i++){
           cache[arr[i]].push_back(i);
        }
    }
    int lower_bound(vector<int> &nums,int tar){
        //寻找第一个大于等于tar的值
        //左闭右开
        //循环不变量 nums[left-1]<tar nums[right]>=tar
        int left=0,right=nums.size();
        int mid;
        while(left<right){
            mid=left+(right-left)/2;
            if(nums[mid]<tar)
                left=mid+1;
            else
                right=mid;
        }
        return left<nums.size()?left:nums.size();
    }
    int query(int left, int right, int value) {
        vector<int> &temp=cache[value];
        int left_index=lower_bound(temp,left);//>=left
        int right_index=lower_bound(temp,right+1);//>=right+1即为>right
        return right_index-left_index;
    }
};

```

### 统计公平数对

> 思路：遍历`nums`数组，计算每个`nums[j]`的边界。即`lower-nums[i]<=nums[j]<=upper-nums[i]`。
>
> 找第一个大于等于`lower-nums[i]`的位置。找第一个`>uppper-nums[i]`的位置。（从零开始，数对值刚好为相减作差。）
>
> 二者相减 即为符合要求数对

```c++
class Solution {
public:
    int lower_bound(vector<int> &nums,int tar,int left,int right){
        //定义找到第一个大于等于tar的位置
        //左闭右闭区间
        //循环不变量nums[left-1]<tar  nums[right+1]>=tar
        
        int mid;
        int temp=left;
        while(left<=right){
            mid=left+(right-left)/2;
            if(nums[mid]<tar)
                left=mid+1;
            else
                right=mid-1;
        }
        return left<nums.size()&&left>=temp?left:nums.size();
    }
    long long countFairPairs(vector<int>& nums, int lower, int upper) {
        sort(nums.begin(),nums.end());
        long long res=0;
        int len=nums.size()-1;
        for(int i=0;i<=len;i++){
            //左闭右开的
            int left_wall=lower-nums[i];
            int right_wall=upper-nums[i];
            int left_index=lower_bound(nums,left_wall,i+1,len);//>=left_wall
            int right_index=lower_bound(nums,right_wall+1,i+1,len)-1;
            res+=right_index-left_index+1;
        }
        return res;
    }
};


class Solution {
public:
    long long countFairPairs(vector<int>& nums, int lower, int upper) {
        long long res=0;
        sort(nums.begin(),nums.end());
        for(int i=0;i<nums.size();i++){
            auto left_index=lower_bound(nums.begin()+i+1,nums.end(),lower-nums[i]);
            auto right_index=upper_bound(nums.begin()+i+1,nums.end(),upper-nums[i]);
            res+=right_index-left_index;
        }
        return res;
    }
};
```

### 删除数对后的最小数组长度

> 思路：此题运用数学知识分类讨论。假设 $ x $ 出现次数最多，其出现次数为 $ \textit{maxCnt} $。
>
> 分类讨论：
>
> 1. 如果 $ \textit{maxCnt} \cdot 2 > n $，其余所有 $ n - \textit{maxCnt} $ 个数都要与 $ x $ 消除，所以最后剩下 $ \textit{maxCnt} \cdot 2 - n $ 个数。
> 2. 如果 $ \textit{maxCnt} \cdot 2 \le n $ 且 $ n $ 是偶数，那么可以把其余数消除至剩下 $ \textit{maxCnt} $ 个数，然后再和 $ x $ 消除，最后剩下 $ 0 $ 个数。
> 3. 如果 $ \textit{maxCnt} \cdot 2 \le n $ 且 $ n $ 是奇数，同上，最后剩下 $ 1 $ 个数。
>
> 所以本题核心是计算 $ \textit{maxCnt} $，这可以遍历一遍 $ \textit{nums} $ 算出来。

```c++
class Solution {
public:
    int minLengthAfterRemovals(vector<int>& nums) {
        int len=nums.size();
        //计算cx_max
        auto left=lower_bound(nums.begin(),nums.end(),nums[len/2]);
        auto right=upper_bound(nums.begin(),nums.end(),nums[len/2]);
        int sub=right-left;
        return max(2*sub-len,len%2);
    }
};
```

### 基于时间的键值存储（不熟悉）

> 思路：采用一个`map<string,vector<pair<int,string>>>`存储。由于时间戳都是升序的。因此可以采用二分法。取值时。判断`vector`是否为空，如果为空，则说明没有值，返回`“”`，利用`upper_bound`找到第一个大于`timestamp`的值`inedx`。如果`index`为`0`说明，没有大于`timestamp`的值 返回`“”`否则返回`index-1`对应`pair`的`value`

```c++
class TimeMap {
public:
    // 自定义 upper_bound 函数，查找第一个大于 tar 的位置
    int upper_bound1(vector<pair<int, string>> &nums, int tar) {
        // 定义左闭右开区间
        // 循环不变量: nums[left-1].first <= tar, nums[right].first > tar
        int left = 0, right = nums.size();
        int mid;
        while (left < right) {
            mid = left + (right - left) / 2;
            if (nums[mid].first > tar)
                right = mid; // 右边界收缩
            else
                left = mid + 1; // 左边界收缩
        }
        // 如果 left 不为 0，返回 left，表示找到的位置，否则返回 -1
        return left != 0 ? left : -1;
    }

    // 使用 map 存储 key 对应的 vector，其中包含时间戳和值对
    map<string, vector<pair<int, string>>> my_map;

    // 构造函数
    TimeMap() {}

    // 设置 key, value 和 timestamp
    void set(string key, string value, int timestamp) {
        my_map[key].emplace_back(timestamp, value);
    }

    // 获取指定 key 和 timestamp 对应的值
    string get(string key, int timestamp) {
        vector<pair<int, string>> &ve = my_map[key];
        // 如果 vector 为空，返回空字符串
        if (ve.size() == 0)
            return "";
        // 查找第一个大于 timestamp 的位置
        int index = upper_bound1(ve, timestamp);
        // 如果 index 为 -1，表示没有找到合适的位置，返回空字符串
        if (index == -1) {
            return "";
        }
        // 返回 index-1 对应的值，因为 index 是第一个大于 timestamp 的位置
        return ve[index - 1].second;
    }
};

/**
 * Your TimeMap object will be instantiated and called as such:
 * TimeMap* obj = new TimeMap();
 * obj->set(key, value, timestamp);
 * string param_2 = obj->get(key, timestamp);
 */

```

### 快照数组（不太熟悉）

> 思路：利用哈希表+二分查找实现。.哈希表存储以数组下标为键，`vector<pair<int,int>>`为值。代表数组位置插入的历史值。当调用`get`函数时。只需要找出，最大的那一个`<=snap`的`pair`等价于查找第一个`>snap`的位置减一

```c++
class SnapshotArray {
public:
    unordered_map<int,vector<pair<int,int>>> mp;//键为数组下标，值为历史修改数据。
    int snap_id=0;
    SnapshotArray(int length) {
         for(int i=0;i<length;i++)
            mp[i]=vector<pair<int,int>>();
    }
    int upper_bound(vector<pair<int,int>> &nums,int tar){
        //查找第一个大于x的数
        //定义区间左臂右开
        //循环不变量 nums[left-1]<=tar  nums[right]>tar
        int left=0,right=nums.size();
        int mid;
        while(left<right){
            mid=left+(right-left)/2;
            if(nums[mid].first>tar)
                right=mid;
            else
                left=mid+1;
        }
        return left;
    }
    void set(int index, int val) {
       mp[index].emplace_back(snap_id, val);
    }
    
    int snap() {
        return snap_id++;
    }
    
    int get(int index, int snap) {
        vector<pair<int,int>> &history = mp[index];
        if (history.empty()) return 0;

        int pos = upper_bound(history, snap);
        if (pos == 0) return 0;
        return history[pos-1].second;
    }
};

/**
 * Your SnapshotArray object will be instantiated and called as such:
 * SnapshotArray* obj = new SnapshotArray(length);
 * obj->set(index,val);
 * int param_2 = obj->snap();
 * int param_3 = obj->get(index,snap_id);
 */
```

### 绝对值差值和（有点懵）

> 思路：使用排序后的 `copy_nums1` 数组和二分查找，计算初始的绝对差值总和，并通过寻找最接近值来优化最大绝对差值替换，最终计算最小化后的绝对差值总和。
>
> 解法：
>
> 1. **复制和排序**：首先复制 `nums1` 到 `copy_nums1`，并对 `copy_nums1` 进行排序。这样可以在 `copy_nums1` 中使用二分查找来快速找到最接近 `nums2[i]` 的值。
>
> 2. **计算初始总绝对差值和**：遍历 `nums1` 和 `nums2`，计算每对元素的绝对差值并累加到 `total_diff` 中，并对 `total_diff` 取模以防止溢出。
> 3. **寻找最大差值和最优替换值**：遍历 `nums1` 和 `nums2`，对于每对元素，计算当前的绝对差值 `current_diff`。然后使用 `lower_bound` 在 `copy_nums1` 中找到大于等于 `nums2[i]` 的位置 `pos`。
> 4. **更新最大差值**：对于找到的位置 `pos`，分别计算替换 `copy_nums1[pos]` 和 `copy_nums1[pos-1]` 后的差值，并更新 `max_diff`。
> 5. **计算最终结果**：将总绝对差值和 `total_diff` 减去最大差值 `max_diff`，并确保结果非负并取模，返回最终结果。

```c++
class Solution {
public:
    int lower_bound(vector<int>& nums1, int tar) {
        // 找到第一个大于等于 tar 的位置
        int left = 0, right = nums1.size();
        int mid;
        while (left < right) {
            mid = left + (right - left) / 2;
            if (nums1[mid] < tar)
                left = mid + 1;
            else
                right = mid;
        }
        return left;
    }

    int upper_bound(vector<int>& nums1, int tar) {
        // 找到第一个大于 tar 的位置
        int left = 0, right = nums1.size();
        int mid;
        while (left < right) {
            mid = left + (right - left) / 2;
            if (nums1[mid] <= tar)
                left = mid + 1;
            else
                right = mid;
        }
        return left;
    }

    int minAbsoluteSumDiff(vector<int>& nums1, vector<int>& nums2) {
        vector<int> copy_nums1(nums1);
        sort(copy_nums1.begin(), copy_nums1.end());
        int total_diff = 0, max_diff = 0, n = nums1.size();
        const int MOD = 1000000007;

        // 计算初始的总绝对差值和
        for (int i = 0; i < n; ++i) {
            total_diff = (total_diff + abs(nums1[i] - nums2[i])) % MOD;
        }

        // 寻找最大的差值和最优替换值
        for (int i = 0; i < n; ++i) {
            int current_diff = abs(nums1[i] - nums2[i]);
            int pos = lower_bound(copy_nums1, nums2[i]);
            
            // 检查 pos 和 pos-1 的最优差值
            if (pos < n) {
                max_diff = max(max_diff, current_diff - abs(copy_nums1[pos] - nums2[i]));
            }
            if (pos > 0) {
                max_diff = max(max_diff, current_diff - abs(copy_nums1[pos - 1] - nums2[i]));
            }
        }

        return (total_diff - max_diff + MOD) % MOD;
    }
};


//官解
class Solution {
public:
    static constexpr int mod = 1'000'000'007;

    int minAbsoluteSumDiff(vector<int>& nums1, vector<int>& nums2) {
        vector<int> rec(nums1);
        sort(rec.begin(), rec.end());
        int sum = 0, maxn = 0;
        int n = nums1.size();
        for (int i = 0; i < n; i++) {
            int diff = abs(nums1[i] - nums2[i]);
            sum = (sum + diff) % mod;
            int j = lower_bound(rec.begin(), rec.end(), nums2[i]) - rec.begin();
            if (j < n) {
                maxn = max(maxn, diff - (rec[j] - nums2[i]));
            }
            if (j > 0) {
                maxn = max(maxn, diff - (nums2[i] - rec[j - 1]));
            }
        }
        return (sum - maxn + mod) % mod;
    }
};
```

### 在线选举（不会）

> 思路：在初始化时计算出每个时刻的领先候选人，并存储在 `tops` 数组中。对于每个查询时间 `t`，使用二分查找（通过 `upper_bound`）找到小于等于 `t` 的最大时间点，直接返回该时间点的领先候选人

```c++
class TopVotedCandidate {
public:

    vector<int> times;
    vector<int> tops;
    int upper_bound(vector<int> &nums,int target){
        //寻找第一个大于target的位置
        //定义左闭右开区间
        //定义循环不变量。nums[left-1]<=tar  nums[right]>tar
        int left=0,right=nums.size();
        int mid;
        while(left<right){
            mid=left+(right-left)/2;
            if(nums[mid]>target)
                right=mid;
            else
                left=mid+1;
        }
        return left;
    }
    TopVotedCandidate(vector<int>& persons, vector<int>& times) {
        //定义一个map用于存放 候选人选票
        unordered_map<int,int> voteCounts;
        voteCounts[-1]=-1;
        int val=-1;
        for(auto &person:persons){
            voteCounts[person]++;
            if(voteCounts[person]>=voteCounts[val]){
                //如果当前时刻的候选人选票大于最多一个
                //更新val=当前候选人
                val=person;
            }
            tops.emplace_back(val);
        }
        //更新完成后将，times数组赋值为类成员变量times
        this->times=times;
    }
    
    int q(int t) {
        int index=upper_bound(times,t)-1;
        return tops[index];
    }
};
```

### [LCP 08. 剧情触发时间](https://leetcode.cn/problems/ju-qing-hong-fa-shi-jian/)

> 思路：由于资源是逐渐递增的，因此可以使用二分法。创建三个`vector`数组存放资源。然后遍历`require`数组。使用二分法在每个资源数组**查找第一个大于等于资源**的位置。一共三个位置。取最大的一个。如果最大一个小于数组长度，即是合法的。否则想`res`数组中插入-1
>
> 解法：
>
> 1. **前缀和数组**：
>
>    通过前缀和数组 `c`, `r`, `h` 来记录每一天的累计资源数，方便后续的查询。
>
> 2. **二分查找**：
>
> 使用二分查找 `lower_bound` 函数来找到满足要求的最早天数，提高效率。
>
> 3. **结果计算**：
>
> 对于每个要求，找出三个累计资源数组中满足要求的最早天数的最大值，确保在三种资源都满足要求的最早天数。
>
> 4. **返回结果**：
>
> 如果找到的最大天数在有效范围内，则返回该天数，否则返回 `-1`。

```c++
class Solution {
public:
    // 自定义的 lower_bound 函数，寻找第一个大于等于 tar 的位置
    int lower_bound(vector<int> &nums, int tar) {
        int left = 0, right = nums.size();  // 设置搜索区间为 [left, right)
        int mid;
        while (left < right) {  // 当 left == right 时终止循环
            mid = left + (right - left) / 2;  // 计算中间位置
            if (nums[mid] < tar)  // 如果中间元素小于目标值
                left = mid + 1;  // 将搜索区间缩小到 [mid + 1, right)
            else  // 如果中间元素大于等于目标值
                right = mid;  // 将搜索区间缩小到 [left, mid)
        }
        return left;  // 返回第一个大于等于 tar 的位置
    }

    vector<int> getTriggerTime(vector<vector<int>>& increase, vector<vector<int>>& requirements) {
        int n = increase.size();
        // 初始化三个资源累积值的数组
        vector<int> c(n + 1, 0), r(n + 1, 0), h(n + 1, 0);
        vector<int> res;  // 保存结果

        // 计算每天的资源累积值
        for (int i = 0; i < n; ++i) {
            c[i + 1] = increase[i][0] + c[i];  // 第 i 天的 c 值
            r[i + 1] = increase[i][1] + r[i];  // 第 i 天的 r 值
            h[i + 1] = increase[i][2] + h[i];  // 第 i 天的 h 值
        }

        // 遍历每个需求
        for (const auto& req : requirements) {
            // 分别找到满足三个资源需求的最早天数
            int index_c = lower_bound(c, req[0]);
            int index_r = lower_bound(r, req[1]);
            int index_h = lower_bound(h, req[2]);
            // 取三个资源需求的最大天数
            int max_index = max({index_c, index_r, index_h});
            // 如果 max_index 小于等于 n，表示可以满足需求，添加到结果中
            // 否则表示无法满足需求，添加 -1 到结果中
            res.push_back(max_index <= n ? max_index : -1);
        }

        return res;  // 返回结果
    }
};

```

### [使结果不超过阈值的最小除数(不太熟悉)](https://leetcode.cn/problems/find-the-smallest-divisor-given-a-threshold/)

> 思路：选择一个除数`d`,当除数增大或减小时。他们的除法和同样也在减小或增大。因此除法和`total`单调。可以使用二分法。选择一个边界。左边界为1，这是最小的正整数。右边界为数组中最大数`M`。因为如果为`M`，那么他们的除法和刚好等于数组长度`nums.size()`。而`threshold`一定是大于等于`nums.size()`。
>
> 解法：
>
> 如果 `total > threshold`，那就说明我们选择的 `d` 不满足要求。根据单调性，减小 `d` 的值会增大 `total `的值，那么区间 [l, d'] 中不可能存在满足要求的除数，因此我们可以在区间 `(d, r]` 中继续进行二分查找。
>
> 如果 `total <= threshold`，那就说明我们选择的 `d` 满足要求。由于题目中要求除数尽可能小，因此我们可以忽略区间 `(d, r]`，而在区间 `[l, d) `中继续进行二分查找。
>
> 

```C++
class Solution {
public:
    int smallestDivisor(vector<int>& nums, int threshold) {
        // 初始化二分查找的左边界和右边界
        int left = 1, right = *max_element(nums.begin(), nums.end());
        int mid;

        // 二分查找，直到 left > right
        //定义为左闭右闭区间
        //查找第一个使得除法和<=threshol的值
        //定义循环不变量为 当total' > threshold  移动left  total' <= threshold  移动right
        while (left <= right) {
            mid = left + (right - left) / 2; // 计算中间值
            int sum = 0;
            for (int num : nums) {
                // 将每个元素除以 mid 后向上取整，然后累加
                sum += (num + mid - 1) / mid; // 等价于 ceil(num / mid)
            }
            // 如果当前的 sum 大于阈值 threshold，说明 mid 太小，需要增大
            if (sum > threshold) {
                left = mid + 1;
            }
            // 否则 mid 可能是结果的一部分，尝试更小的值
            else {
                right = mid - 1;
            }
        }
        
        // 返回满足条件的最小除数
        return left;
    }
};

```

### 



```c++
class Solution {
public:
    long long minimumTime(vector<int>& time, int totalTrips) {
        // 定义二分查找的左右边界
        long long left = 1;
        long long right = (long long)totalTrips * time[0]; // 最大时间假设是最慢的车完成所有的旅行
        
        long long mid = 0;
        
        // 开始二分查找
		// 定义为左闭右闭区间 [left, right]
		// 循环不变量为：在区间 [left, right] 内寻找满足条件的最小值，
        //其中 left-1 左边的 sum 小于 totalTrips，right 右边的 sum 大于等于 totalTrips
        while (left <= right) {
            // 计算中间值
            mid = left + (right - left) / 2;
            
            // 计算在 mid 时间内所有公交车能完成的总旅行次数
            long long sum = 0;
            for (int i = 0; i < time.size(); i++) {
                sum += mid / time[i]; // 当前时间内，每辆车能完成的旅行次数
            }
            
            // 如果在 mid 时间内完成的总旅行次数小于 totalTrips，则说明时间太短
            if (sum < totalTrips) {
                left = mid + 1; // 增加左边界，缩小搜索范围
            } else {
                // 否则，说明时间足够或者有富余，尝试更短的时间
                right = mid - 1; // 减少右边界，缩小搜索范围
            }
        }
        // 返回完成所有旅行的最短时间
        return left;
    }
};

```

### [1870. 准时到达的列车最小时速(边界问题没搞清楚)](https://leetcode.cn/problems/minimum-speed-to-arrive-on-time/)

> 思路：由于需要按次序坐车。题目也给出了最大的答案范围。同时答案为正整数。因此可以考虑使用二分法进行遍历。得出最小的时速。每次二分需要一个`check`函数检查当前`speed`是否符合要求。如果符合要求，继续向左移动。求得最小。不符合说明speed过小。  `left`向右移动。最终位置为`right=left+1`返回`right`即可。
>
> 解法：
>
> **二分查找确定最小速度**:
>
> 1. 初始化边界：
>
>    - 左边界 `left` 为 1（最小正整数速度）。
>
>    - 右边界 `right` 为 `1e7 + 1`（一个足够大的值，假设的最大速度）。
>
> 2. 二分查找过程：
>
>    - 计算中间值 `mid`。
>
>    - 定义一个辅助函数 `check` 来检查在速度 `mid` 下是否能在给定的 `hour` 时间内完成所有行程。
>
>    - 在 `check` 函数中，计算每个车站的行程时间，如果不是最后一站则向上取整，总时间不超过 `hour` 则满足条件。
>
>    - 如果当前速度 `mid` 满足条件，则收缩右边界 `right` 为 `mid`，否则收缩左边界 `left` 为 `mid`。
>
> 3. 结束条件：
>    - 当 `left+1` 与 `right` 相遇时，返回右边界 `right`，即为所求的最小速度。

```c++
class Solution {
public:
    // 思路：最小正整数为 1，因此左边界为 1，而右边界为 1e7。
    
    // 检查当前得出的速度 speed 是否符合题目要求
    bool check(vector<int> &dist, int speed, double hour) {
        double sum = 0.0;  // 总时间
        for (int i = 0; i < dist.size(); i++) {
            if (i != dist.size() - 1) {
                // 如果不是最后一站需要向上取整
                sum += (dist[i] + speed - 1) / speed; // (dist[i] + speed - 1) / speed 是向上取整的一个技巧
            } else {
                sum += (double)dist[i] / speed; // 最后一站的时间可以是小数
            }
        }
        return sum <= hour; // 检查总时间是否小于等于给定的 hour
    }

    int minSpeedOnTime(vector<int>& dist, double hour) {
        // 采用左开右开区间
        int left = 0, right = 1e7 + 1; // 定义左边界和右边界
        int mid;
        
        // 如果车站数量大于小时数 + 1，返回 -1
        if (dist.size() >= hour + 1) {
            return -1;
        }
        // 二分查找最小速度
        while (left + 1 < right) {
            mid = left + (right - left) / 2; // 计算中间值
            if (check(dist, mid, hour)) {
                right = mid; // 如果满足条件，则收缩右边界
            } else {
                left = mid; // 否则收缩左边界
            }
        }
        
        return right; // 返回最小速度
    }
};

```

### [1011. 在 D 天内送达包裹的能力](https://leetcode.cn/problems/capacity-to-ship-packages-within-d-days/)

> 思路：找出粗略的边界一个一个试。如果当前的`w(船的重量)`运送完所有货物需要时间大于`days`。说明当前`w`太小了。需要向右收缩区间。如果大于等于`days`说明符合要求。继续向左边收缩边界查找最小的`w`。
>
> 解法：
>
> 1. 粗略边界：船的重量一定大于0，最大重量一定小于最大的包裹乘数组长度。
>
> 2. 精确边界：船的运载能力不可能小于最大的包裹。同时最大不可能超过包裹重量之和(一天就完成)
>
> 3. **初始化边界**：
>
>    - 设定船只运载能力的下界 `left` 为最重的包裹重量，因为船的容量不能小于任何一个包裹。
>    - 设定船只运载能力的上界 `right` 为所有包裹重量的总和，因为在最极端的情况下，可能需要一天运送完所有包裹。
>
>    **二分查找**：
>
>    - 使用二分查找来寻找最小的船只运载能力：
>      - 计算中间值 `mid` 作为当前尝试的船只运载能力。
>      - 使用辅助函数 `check` 来判断在当前运载能力 `mid` 下，是否可以在规定的天数内运送完所有包裹。
>      - 如果可以，则尝试更小的运载能力（调整 `right`）；如果不可以，则需要更大的运载能力（调整 `left`）。
>
>    **辅助函数 `check`**：
>
>    - 模拟运输过程，计算在当前运载能力 `tar` 下，需要多少天才能运送完所有包裹。
>    - 如果天数超过规定的 `days`，返回 `false`；否则返回 `true`。
>
>    **最终结果：**
>
>    * 由于`left+1==right`采用的左开右开区间。因此返回任意`left+1`或者`right`

```c++
class Solution {
public:
    bool check(vector<int> &weights, int tar, int days) {
        int need_days = 1; // 需要的天数，初始化为1天，因为即使是最少的一天也需要1天
        int sum = 0; // 当前一天内已经运输的总重量

        for (int i = 0; i < weights.size(); i++) {
            // 如果当前重量加上新的重量不超过tar
            if (sum + weights[i] <= tar) {
                sum += weights[i]; // 加上当前货物重量
            } else {
                // 否则，需要增加一天
                need_days++;
                if(tar<weights[i])
                    return false;
                sum = weights[i]; // 新的一天开始，sum为当前货物重量

                // 如果所需天数超过了days，直接返回false
                if (need_days > days) {
                    return false;
                }
            }
        }
        return true; // 如果所需天数在允许范围内，返回true
    }

    int shipWithinDays(vector<int>& weights, int days) {
        int left=*max_element(weights.begin(),weights.end())-1;//船的运载能力不可能小于最大的包裹。
        int right=accumulate(weights.begin(), weights.end(), 0)+1;//同时最大不可能超过包裹重量之和(一天就完成)
        int mid;
        while(left+1<right){
            mid=left+(right-left)/2;
            if(check(weights,mid,days)){
                //如果符合，那么继续往左边收缩
                right=mid;
            }else{
                //不符合，说明太小了,需要增大重量
                left=mid;
            }
        }
        return right;
    }
};
```

### [875. 爱吃香蕉的珂珂](https://leetcode.cn/problems/koko-eating-bananas/)

> 思路：如果能在`k=x`的速度内吃掉香蕉。那么`k=x+1,k=x+2......`也一定可以。如果在`k=x`速度不能吃掉香蕉。那么在`k=x-1,k=x-2.....`同样不能。因此可以使用二分法得出最小的速度`k`
>
> 解法：
>
> 1. **确定搜索范围**：
>
>    - 最小速度 `left` 初始化为 1。
>
>    - 最大速度 `right` 初始化为所有香蕉堆中最大的香蕉数。
>
> 2. **二分查找**：
>
>    - 使用二分查找来确定最小的吃香蕉速度：
>
>      - 计算中间速度 `mid`。
>
>      - 使用辅助函数 `check` 判断在速度 `mid` 下是否能在 `h` 小时内吃完所有香蕉：
>
>        1. 如果可以，缩小右边界 `right = mid`。
>
>        2. 如果不可以，增加左边界 `left = mid + 1`。

```c++
class Solution {
public:
    // 检查在给定速度下是否能在规定小时内吃完所有香蕉堆
    bool check(vector<int> &piles, int speed, int h) {
        int sum_h = 0;
        for (int i = 0; i < piles.size(); i++) {
            // 计算当前堆在给定速度下需要多少小时吃完，向上取整
            sum_h += (piles[i] + speed - 1) / speed;
            if (sum_h > h) // 如果总小时数超过规定的h，返回false
                return false;
        }
        return true; // 否则返回true，表示可以在规定时间内吃完所有香蕉
    }

    // 找到能在规定小时内吃完所有香蕉堆的最小速度
    int minEatingSpeed(vector<int>& piles, int h) {
        int left = 0; // 最小的吃香蕉速度，初始化为0，因为速度不能为0
        int right = *max_element(piles.begin(), piles.end()) + 1; // 最大的吃香蕉速度，初始化为最大堆的香蕉数加1
        int mid;
        while (left + 1 < right) {
            mid = left + (right - left) / 2; // 计算中间速度
            if (check(piles, mid, h)) // 如果当前速度mid可行
                right = mid; // 继续往左侧搜索更小的速度
            else
                left = mid; // 否则，增加速度，往右侧搜索更大的速度
        }
        return right; // 返回右侧边界，即找到的最小可行速度
    }
};

```

### [475. 供暖器](https://leetcode.cn/problems/heaters/)

> 思路1：利用排序+二分查找。将`heaters`数组排序，每次找到第一个`>=houses`的位置，以及找到最大一个小于`houses`位置。取二者与`houses`绝对值最小的一个。再跟`res`取最大的一个。
>
> 解法1：
>
> * 如果 `big_index` 超出加热器数组范围，`first` 设为 `INT_MAX`。
>
> * 如果 `less_index` 小于 0，`second` 设为 `INT_MAX`。
>
> 思路2：排序+双指针。核心思想。对于每个房屋，要么用前面的暖气，要么用后面的，二者取近的，得到距离。排序``heaters`数组。 取两个指针`i,j`，`i`遍历房屋。`j`指向上一个房屋的热水器位置。遍历房屋数组。如果上一个热水器到当前房屋距离大于等于下一个热水器到当前房屋距离，将`j++`代表离当前房屋最近的一个热水器。每次遍历与`res`取最大的一个。

```c++
//解法1
class Solution {
public:
    // 自定义 lower_bound 函数，返回第一个大于等于目标值 tar 的索引
    int lower_bound(vector<int> &nums, int tar) {
        // 定义左开右开区间
        // 循环不变量为 nums[left] < tar，nums[right] >= tar
        int left = -1, right = nums.size();
        while (left + 1 < right) {
            int mid = left + (right - left) / 2;  // 计算中间索引
            if (nums[mid] < tar)
                left = mid;  // 若中间值小于目标值，移动左边界
            else
                right = mid;  // 否则，移动右边界
        }
        return right;  // 返回第一个大于等于目标值的索引
    }

    int findRadius(vector<int>& houses, vector<int>& heaters) {
        // 对加热器位置进行排序
        sort(heaters.begin(), heaters.end());
        int res = 0;  // 初始化结果变量

        // 遍历每个房屋，找到离它最近的加热器
        for (int i = 0; i < houses.size(); i++) {
            int house = houses[i];

            // 找到第一个大于等于 house 位置的加热器索引
            int big_index = lower_bound(heaters, house);
            // 找到第一个小于 house 位置的加热器索引
            int less_index = big_index - 1;

            // 计算当前房屋到最近加热器的距离
            // first 是当前房屋到大于等于它位置的加热器的距离
            int first = big_index < heaters.size() ? abs(heaters[big_index] - house) : INT_MAX;
            // second 是当前房屋到小于它位置的加热器的距离
            int second = less_index >= 0 ? abs(heaters[less_index] - house) : INT_MAX;

            // 取最小的距离作为当前房屋到最近加热器的距离
            int temp = min(first, second);
            // 更新最大最小距离
            res = max(temp, res);
        }

        return res;  // 返回结果
    }
};


//解法2
class Solution {
public:
//尝试排序+双指针做法  当前房屋，要么使用前一个热水器覆盖的，不然就是后面热水器覆盖
    int findRadius(vector<int>& houses, vector<int>& heaters) {
        int res=0;
        sort(houses.begin(),houses.end());
        sort(heaters.begin(),heaters.end());
        for(int i=0,j=0;i<houses.size();i++){
            //如果房屋前一个热水器覆盖距离比后面覆盖距离大。那么选择后面小的一个热水器. j表示当前满足前一个房屋覆盖距离的热水器位置
            while(j<heaters.size()-1&&abs(houses[i]-heaters[j])>=abs(heaters[j+1]-houses[i]))
                j++;
            res=max(res,abs(houses[i]-heaters[j]));
        }
        return res;
    }
};
```

### [2594. 修车的最少时间（不会）](https://leetcode.cn/problems/minimum-time-to-repair-cars/)

> 思路：根据计算每个职工在时间`t`内最多可以修理`sqrt(rank/t)`台车。使用二分查找。定义左边界为0右边界为`ranks[0]*car*car`。计算`mid`时间内，一共可以修理多少台车。如果大于`cars`代表满足条件 向左收缩边界寻找最小时间。如果小于`cars`代表不满足条件。需要增大时间。
>
> 解法：
>
> 1. **初始化边界**：
>
>    - `left` 初始为 0，表示最短的修理时间可能为 0。
>
>    - `right` 初始为 `ranks[0] * cars * cars`，表示可能的最长修理时间，这是基于最慢修理工修理所有汽车的时间上限计算的。
>
>    - 将 `right` 强制转换为 `long long` 类型以防止整数溢出。
>
> 2. **二分查找**：
>
>    - 进入循环，当 `left + 1 < right` 时继续执行：
>
>      - 计算中间时间 `mid = left + (right - left) / 2`。
>
>      * 初始化 `sum` 为 0，统计在 `mid` 时间内可以修理的汽车数量。
>
>      * 遍历每个修理工，计算当前修理工在 `mid` 时间内可以修理的汽车数量并累加到 `sum`：
>        - 每个修理工在 `mid` 时间内修理的汽车数量为 `floor(sqrt(mid / ranks[i]))`。
>
>      * 检查 `sum` 是否大于等于需要修理的汽车数量 `cars`：
>        - 如果 `sum >= cars`，说明 `mid` 时间足够修理所有汽车，将 `right` 更新为 `mid` 以收缩搜索区间。
>        - 否则，将 `left` 更新为 `mid` 以增大搜索区间。

```c++
class Solution {
public:
    // 判断在时间 tar 内是否可以修理所有 cars 辆汽车
    bool check(int nums[], long long tar, int cars, int min_r) {
        long long sum = 0;
        for (int i = min_r; i <= 100; i++) {
            sum += (long long) sqrt(tar / i) * nums[i];
            if (sum >= cars)
                return true;
        }
        return false;
    }

    long long repairCars(vector<int>& ranks, int cars) {
        int min_r = ranks[0], cnt[101] = {}; // 数组比哈希表更快
        // 统计每个等级修理工的数量，并找到最小等级
        for (int r : ranks) {
            min_r = min(min_r, r);
            cnt[r]++;
        }
        long long left = 0, right = 1LL * min_r * cars * cars;
        long long mid;
        while (left + 1 < right) {
            mid = left + (right - left) / 2;
            if (check(cnt, mid, cars, min_r))
                right = mid;
            else
                left = mid;
        }
        return right;
    }
};
```

### [1482. 制作 m 束花所需的最少天数](https://leetcode.cn/problems/minimum-number-of-days-to-make-m-bouquets/)

> 思路：由于要求的是所需的最少天数。因此目标很明确。确定最大需要天数，以及最小需要天数。通过二分查找，每次检查当前`days`是否满足要求。若满足要求。向左收缩区间。若不满足要求,向右收缩区间。最后返回`right`即为所求
>
> 解法：
>
> 1. **二分查找**：
>    - 初始化 `left` 为数组中的最小开花时间减1，`right` 为数组中的最大开花时间加1。
>    - 通过二分查找的方法确定最小的满足条件的天数。
> 2. **检查函数** (`check`)：
>    - 遍历 `bloomDay` 数组，计算当前连续符合条件的花朵数 `flowers`。
>    - 当 `flowers` 达到 `k` 时，增加符合条件的子数组计数 `count`，并重置 `flowers`。
>    - 如果遇到不符合条件的花朵，重置 `flowers`。
>    - 返回找到的符合条件的子数组个数是否大于或等于 `m`。
> 3. **二分调整**：
>    - 如果 `check` 函数返回 `true`，表示当前天数可以满足条件，向左收缩 (`right = mid`)。
>    - 否则，向右收缩 (`left = mid`)。
> 4. **结果返回**：
>    - 最终返回 `right`，即最小的满足条件的天数。

```c++
class Solution {
public:
    // 检查在给定天数内是否能找到至少 m 个长度为 k 且所有元素小于等于 days 的连续子数组
    bool check(vector<int>& bloomDay, int m, int k, int days) {
        int count = 0; // 计数器，记录符合条件的子数组个数
        int flowers = 0; // 记录当前连续的符合条件的花朵数
        
        for (int i = 0; i < bloomDay.size(); ++i) {
            // 如果当前花朵开花时间小于等于给定的天数
            if (bloomDay[i] <= days) {
                flowers++;
                // 当连续符合条件的花朵数达到 k
                if (flowers == k) {
                    count++;
                    flowers = 0; // 重置 flowers 以继续检查后面的子数组
                }
            } else {
                // 如果遇到一个花朵开花时间大于给定的天数，重置 flowers
                flowers = 0;
            }
        }
        
        // 返回是否符合条件的子数组个数大于或等于 m
        return count >= m;
    }

    // 找到能使 m 个连续子数组每个长度为 k 且所有元素小于等于 days 的最小天数
    int minDays(vector<int>& bloomDay, int m, int k) {
        // 如果总的花朵数少于 m*k，无法满足条件，返回 -1
        if (bloomDay.size() < (long long)m * k)
            return -1;

        // 初始化二分查找的左右边界
        int left = *min_element(bloomDay.begin(), bloomDay.end()) - 1;
        int right = *max_element(bloomDay.begin(), bloomDay.end()) + 1;
        int mid;

        // 进行二分查找
        while (left + 1 < right) {
            mid = left + (right - left) / 2;
            // 如果 mid 天数满足条件，向左收缩
            if (check(bloomDay, m, k, mid))
                right = mid;
            else
                left = mid; // 否则向右收缩
        }

        // 返回最小的满足条件的天数
        return right;
    }
};

```

### [H 指数 II](https://leetcode.cn/problems/h-index-ii/)

> 思路：查看序列已经是升序，同时检查答案是否具有单调性。如果至少有2篇论文的引用次数≥2,那么也必然有至少1篇论文的引用次数≥1。如果没有4篇论文的引用次数≥4。那么也必然没有5篇论文的引用次数≥5。由此判断答案具有单调性。进行二分答案。
>
> 答案范围是`[0,n]`，探测1是否符合条件。即检查`citations[len-1]>=1`，如果最后一个大于等于1，说明1符合要求，即向右收缩区间，寻找最大值。

```c++
class Solution {
public:
    int hIndex(vector<int>& citations) {
        //检测是否h 为citations[len-tar]>=tar  
        //定义为左闭右开区间
        //循环不变量定义为 left左边一定满足要求，right以及右边一定不满足要求。
        int len=citations.size();
        int left=1,right=len+1;
        int mid;
        while(left<right){
            mid=left+(right-left)/2;
            if(citations[len-mid]>=mid)
                //当前满足要求移动left
                left=mid+1;
            else
                right=mid;
        }
        //由于最终会是left=right-1;一定符合条件
        return left-1;
    }
};
```

### [2226. 每个小孩最多能分到多少糖果](https://leetcode.cn/problems/maximum-candies-allocated-to-k-children/)

> 思路 ：求最大/最小，就先往二分答案上想。查看题目，对答案进行二分。
>
> 解法：开区间写法。同时需要一个`check`函数检查当前的`tar`是否可以成功分配。循环不变量定义为`left以及left左边一定满足要求，right以及right右边一定不满足`
>
> 1. `check==true`执行`left=mid`
> 2. `check==false`执行`right==mid`
>
> 3. 最后由于`left=right-1`返回二者任意一个即可。

```c++
class Solution {
public:
    bool check(vector<int> &candies,long long tar,long long k){
        int len=candies.size();
        long long count=0;
        for(int i=0;i<len;i++){
                //检查可以最多分为几堆tar
                count+=candies[i]/tar;
        }
        return count>=k;
    }
    int maximumCandies(vector<int>& candies, long long k) {
        long long left=0,right=*max_element(candies.begin(),candies.end())+1;
        long long mid;
        //定义为左开右开区间
        //left及左边都符合，right及右边都不符合。
        while(left+1<right){
            mid=left+(right-left)/2;
            if(check(candies,mid,k))
                left=mid;
            else
                right=mid;
        }
        return left;
    }
};
```

### [2982. 找出出现至少三次的最长特殊子字符串 II（不会）](https://leetcode.cn/problems/find-longest-special-substring-that-occurs-thrice-ii/)

> 思路：要找至少三次的最长特殊子字符串，将26个字母在数组中出现的连续长度都记录下来，用一个` vector<int> groups[26]`数组（元素为`vector`）存放。接着遍历数组，具体可以分为以下三种情况
>
> 1. 从最长的特殊子串`（a[0]）`中取三个长度均为` a[0]−2` 的特殊子串。例如示例 1 的 `aaaa `可以取三个 `aa`。
> 2. 或者，从最长和次长的特殊子串`（a[0],a[1]）`中取三个长度一样的特殊子串：
>    * 如果 `a[0]=a[1]`，那么可以取三个长度均为 `a[0]−1 `的特殊子串。
>    * 如果 `a[0]>a[1]`，那么可以取三个长度均为` a[1]` 的特殊子串：从最长中取两个，从次长中取一个。
>    * 这两种情况合并成 `min(a[0]−1,a[1])`。
> 3. 又或者，从最长、次长、第三长的的特殊子串`（a[0],a[1],a[2]）`中各取一个长为 `a[2]` 的特殊子串。

```c++
class Solution {
public:
    int maximumLength(string s) {
        vector<int> groups[26]; // 定义26个vector<int>数组，用于存储每个字符连续出现的长度

        // 统计字符长度
        int cnt = 0, n = s.size();
        for (int i = 0; i < s.size(); i++) {
            cnt++; // 增加当前字符计数
            // 如果当前字符与下一个字符不同，或者已经是最后一个字符
            if (i == n - 1 || s[i] != s[i + 1]) {
                groups[s[i] - 'a'].emplace_back(cnt); // 将当前字符的连续出现次数存储到对应的vector中
                cnt = 0; // 重置计数器
            }
        }

        int ans = 0; // 初始化结果变量

        // 对每个字符的连续出现长度进行处理
        for (auto& a : groups) {
            if (a.empty())
                continue; // 如果当前vector为空，则跳过
            std::ranges::sort(a, std::greater<>()); // 对每个vector进行降序排序
            a.push_back(0); // 添加0以便处理最小可能长度(如果只有一个最长子串，后面不好统一处理)
            a.push_back(0); // 再次添加0
            // 计算最大特殊子字符串长度，并更新结果
            ans = std::max({ans, a[0] - 2, std::min(a[0] - 1, a[1]), a[2]});
        }

        return ans ? ans : -1; // 返回结果，如果结果为0，则返回-1
    }
};

```

### [2576. 求出最多标记下标](https://leetcode.cn/problems/find-the-maximum-number-of-marked-indices/)

> 思路：将数组排序，要想实现`k`组配对，那么必须`nums[0]` 肯定要匹配 `nums[n−k]`,否则的话，不可能配对成功。
>
> 思路2：排序+双指针。将数组排序后。从`i=0`开始，`j=(n+1)/2`开始，前半段和后半段元素个数相同。满足条件时将`i++`。最后返回`i*2`

```c++
class Solution {
public:
    // 检查是否可以找到k对满足条件的配对
    bool check(vector<int> &nums, int k) {
        int left = 0, right = nums.size() - k; // 初始化两个指针
        while (left < k && right < nums.size()) {
            if (nums[left++] * 2 <= nums[right++]) // 检查左指针的值的两倍是否小于等于右指针的值
                continue; // 如果满足条件，继续检查下一对
            return false; // 如果不满足条件，返回false
        }
        return true; // 如果所有对都满足条件，返回true
    }
    int maxNumOfMarkedIndices(vector<int>& nums) {
        sort(nums.begin(), nums.end()); // 对数组进行排序
        int left = 0, right = nums.size() / 2 + 1; // 初始化左边界和右边界
        // 定义为左开右开区间，循环不变量为left左边及left一定满足，right及右边一定不满足
        int mid;
        while (left + 1 < right) {
            mid = left + (right - left) / 2; // 计算中间值
            if (check(nums, mid)) // 检查当前mid是否满足条件
                left = mid; // 如果满足，将left更新为mid
            else
                right = mid; // 如果不满足，将right更新为mid
        }
        return left * 2; // 返回最大配对数乘以2
    }
};


//双指针
class Solution {
public:
    int maxNumOfMarkedIndices(vector<int>& nums) {
        sort(nums.begin(), nums.end()); // 对数组进行排序
        int i = 0, n = nums.size(); // 初始化左指针i和数组长度n
        for (int j = (n + 1) / 2; j < n; j++) { // 初始化右指针j为数组中间位置，遍历到数组末尾
            if (nums[i] * 2 <= nums[j]) // 检查左指针i指向的值的两倍是否小于等于右指针j指向的值
                i++; // 如果满足条件，移动左指针
        }
        return i * 2; // 返回最大配对数乘以2
    }
};

```

### [1898. 可移除字符的最大数目](https://leetcode.cn/problems/maximum-number-of-removable-characters/)

> 思路：如果移除前`k+1`个下标满足，那么移除前`k`个依然满足，如果移除`k`个不满足，那么移除`k+1`个任然不满足。
>
> 解法：
>
> **二分查找**：
>
> - 我们使用二分查找来猜测最多可以移除的字符数量。
> - 定义 `left` 为当前可以满足的移除字符数量的最大值，`right` 为超过这个数量一定不能满足的值。
>
> **辅助函数 `check`**：
>
> - 这个函数用于判断在移除了前 `k` 个字符后，字符串 `p` 是否仍然是 `s` 的子序列。
> - 用一个布尔数组 `removed` 标记哪些字符被移除。
> - 遍历 `s`，检查未移除的字符是否可以按顺序匹配到 `p`。
>
> **二分搜索过程**：
>
> - 我们不断调整 `left` 和 `right`，缩小区间范围。
> - 每次取中间值 `mid`，调用 `check` 函数判断在移除 `mid` 个字符后，`p` 是否仍然是 `s` 的子序列。
> - 根据判断结果调整 `left` 和 `right` 的值，直到确定最多可以移除的字符数量。

```c++
class Solution {
public:
    // 检查 p 是否仍然为 s 的子序列
    bool check(const string& s, const string& p, const vector<int>& remove_index, int k) {
        int i = 0, j = 0;
        int lens = s.size(), lenp = p.size();
        vector<bool> removed(lens, false);

        // 标记需要移除的字符
        for (int index = 0; index < k; ++index) {
            removed[remove_index[index]] = true;
        }

        // 遍历 s 来匹配 p
        while (i < lens && j < lenp) {
            if (!removed[i] && s[i] == p[j]) {
                j++;
            }
            i++;
        }
        return j == lenp;
    }

    int maximumRemovals(string s, string p, vector<int>& removable) {
        // 二分猜答案，左开右开区间。循环不变量 left及left左边一定满足，right及right右边一定不满足。
        int left = -1, right = removable.size() + 1;
        int mid;
        while (left +1< right) {
            mid = left + (right - left) / 2;
            if (check(s, p, removable, mid)) {
                left = mid;
            } else {
                right = mid;
            }
        }
        return left;
    }
};

```

### [1802. 有界数组中指定下标处的最大值](https://leetcode.cn/problems/maximum-value-at-a-given-index-in-a-bounded-array/)

> 思路：根据题目，如果`index`处为`tar`时满足目标，那么`tar-1`同样满足目标，如果`tar`不满足结果，那么`tar+1`也不满足结果。由此可以使用二分法进行猜答案。
>
> 解法：定义为左闭右闭区间，循环不变量为 `left`左边一定满足条件，`right`右边一定不满足。 
>
> 1. **二分查找范围确定：** 使用二分查找来确定满足条件的最大值。左边界 `left` 初始化为1，右边界 `right` 初始化为 `maxSum`。
> 2. **检查函数 `check`：** 定义了一个函数 `check`，用来判断给定的 `tar` 是否满足条件。`tar` 表示以第 `index` 个元素为中心，左右两侧等差数列的长度，使得总和不超过 `maxSum`。
> 3. **计算等差数列和：**
>    - 如果 `tar > index`，表示左侧可以完整形成等差数列，计算左侧等差数列的和。
>    - 否则，左侧有一部分是从1开始递增的。
>    - 同样地，计算右侧等差数列的和。
> 4. **二分查找执行：** 根据 `check` 函数的返回值调整 `left` 和 `right` 的值，直到找到满足条件的最大值。
> 5. **返回结果：** 返回满足条件的最大值。



```c++
class Solution {
public:
    // 检查给定的 tar 值是否满足条件
    bool check(int n, int index, int maxSum, int tar) {
        long long sum = tar;  // 将 tar 初始为 sum

        // 计算左边部分的和
        if (tar > index) {
            // 如果 tar 大于左边的元素个数 index，则左边部分是一个完整的等差数列
            sum += (long long)(tar - 1 + tar - index) * index / 2;
        } else {
            // 如果 tar 小于或等于左边的元素个数 index，则左边部分有一部分是 1
            sum += (long long)(tar - 1) * tar / 2 + (index - tar + 1);
        }

        // 计算右边部分的和
        int rightCount = n - index - 1; // 右边的元素个数
        if (tar > rightCount) {
            // 如果 tar 大于右边的元素个数 rightCount，则右边部分是一个完整的等差数列
            sum += (long long)(tar - 1 + tar - rightCount) * rightCount / 2;
        } else {
            // 如果 tar 小于或等于右边的元素个数 rightCount，则右边部分有一部分是 1
            sum += (long long)(tar - 1) * tar / 2 + (rightCount - tar + 1);
        }
        // 检查总和是否小于或等于 maxSum
        return sum <= maxSum;
    }
    int maxValue(int n, int index, int maxSum) {
        int left = 1, right = maxSum;  // 初始化二分查找的边界
        int mid;
        // 进行二分查找
        while (left <= right) {
            mid = left + (right - left) / 2;
            if (check(n, index, maxSum, mid)) {
                left = mid + 1;  // 如果 mid 满足条件，则 mid 可能还可以更大
            } else {
                right = mid - 1;  // 如果 mid 不满足条件，则 mid 需要变小
            }
        }
        return left - 1;  // 返回满足条件的最大值
    }
};

```

### [1642. 可以到达的最远建筑（待优化）](https://leetcode.cn/problems/furthest-building-you-can-reach/)

> 思路：记录当前小于总砖块`bricks`最大的`k`个值，当需要砖块数量`sum`大于总砖块数量，查看是否有梯子，没有返回`false`，有的话使用一次梯子，移除当前需要最大的砖块值。如果能够到达`tar`说明能够到达。否则不能。明显答案具有二义行。能够对于答案进行二分法。
>
> 思路2：使用堆进行优化，维护一个最小堆，堆的大小为梯子的大小，如果超过了，那么就需要使用砖块，如果砖块不够，说明无法到达当前位置。返回上一个位置即可。

```c++
class Solution {
public:
    bool check(vector<int> &nums, int bricks, int ladders, int tar) {
        // 检查能否到达tar位置。记录当前需要的最大砖块。如果综合>bricks 使用一个梯子。最后检查能否到达tar
        int sum = 0;
        multiset<int,greater<int>> max_bricks;
        
        for (int i = 1; i <= tar; i++) {
            if (nums[i] <= nums[i-1]) {
                continue;
            } else {
                int diff = nums[i] - nums[i-1];
                sum += diff;
                // 记录砖块数量
                max_bricks.insert(diff);
                
                if (ladders == 0 && sum > bricks) {
                    return false;
                }
                
                if (sum > bricks) {
                    // 使用一次梯子
                    ladders--;
                    sum -= *max_bricks.begin(); // 使用最大的砖块
                    max_bricks.erase(max_bricks.begin());
                }
            }
        }
        return true;
    }
    int furthestBuilding(vector<int>& heights, int bricks, int ladders) {
        //左开右开区间
        //循环不变量为left及left左边一定满足要求，right及right右边一定不满足
        int left=-1,right=heights.size();
        int mid;
        while(left+1<right){
            mid=left+(right-left)/2;
            if(check(heights,bricks,ladders,mid))
                //满足要求，向右边收缩
                left=mid;
            else
                right=mid;
        }
        return left;
    }
};


//思路2:
class Solution {
public:
    int furthestBuilding(vector<int>& heights, int bricks, int ladders) {
        priority_queue<int,vector<int>,greater<int>> heap;

        for(int i=1;i<heights.size();i++){
            int diff=heights[i]-heights[i-1];
            if(diff>0)heap.push(diff);

            //检查是否超过了
            if(heap.size()>ladders){
                //只能使用砖块
                bricks-=heap.top();
                heap.pop();
            }
            if(bricks<0)
                return i-1;
        }
        return heights.size()-1;
    }
};
```

### [2861. 最大合金数（第一次未AC）](https://leetcode.cn/problems/maximum-number-of-alloys/)

> 思路：通过二分进行猜测答案，不需要找出具体由哪个机器进行制造，仅需要知道当前对应的`tar`份合金是否能够被制造出。如果能则向右收缩，不能则向左收缩。

```c++
class Solution {
public:
    // 检查是否能使用指定机器在预算内生产出 max_count 个合金
    bool check(vector<vector<int>>& composition, int budget, int n, int max_count, int machine, vector<int>& cost, vector<int>& stock) {
        long long sum = 0;  // 用于累计所需的总成本
        for (int i = 0; i < n; i++) {
            // 计算生产 max_count 个合金所需的每种金属数量减去当前库存
            long long required = static_cast<long long>(max_count) * composition[machine][i] - stock[i];
            if (required > 0) {
                sum += required * cost[i];  // 累计需要购买的金属成本
                if (sum > budget) return false;  // 如果超过预算，直接返回 false
            }
        }
        return sum <= budget;  // 返回是否在预算内
    }

    // 计算最多能生产多少合金
    int maxNumberOfAlloys(int n, int k, int budget, vector<vector<int>>& composition, vector<int>& stock, vector<int>& cost) {
        int left = 0, right = *max_element(stock.begin(), stock.end()) + budget + 1;  // 二分查找的范围
        int mid;
        
        // 二分查找，找出能生产的最大合金数
        while (left + 1 < right) {  // 左开右开区间
            mid = left + (right - left) / 2;  // 计算中间值
            bool can_make = false;
            for (int i = 0; i < k; i++) {
                // 检查是否存在至少一台机器可以在预算内生产出 mid 个合金
                if (check(composition, budget, n, mid, i, cost, stock)) {
                    can_make = true;
                    break;
                }
            }
            if (can_make)
                left = mid;  // 如果可以生产，左边界移动到 mid
            else
                right = mid;  // 否则，右边界移动到 mid
        }
        return left;  // 返回能生产的最大合金数
    }
};
```

### [3143. 正方形中的最多点数](https://leetcode.cn/problems/maximum-points-inside-the-square/)

> 思路：根据题目提示，包含点 `(x, y) `的正方形的最小边长为 `max(abs(x), abs(y)) * 2。`因此我们可以二分答案，左边界为`-1`，右边界为最大边长绝对值+1。如果一个点的最大边长小于`tar`说明他可以被包含在正方形内，同时我们维护一个`set`集合，用来判断是否有相同标记的点。最后在`check`函数中记录正方形中点的个数。同时判断是否有重复标签。
>
> ### 思路总结
>
> 1. **计算曼哈顿距离**：
>    - 对于每个点，计算其到原点的最大曼哈顿距离，并将这些距离存储在一个向量中。这一步的目的是确定每个点到原点的距离，以便后续判断点是否在正方形内。
> 2. **二分查找最大边长**：
>    - 初始化二分查找的范围，`left` 从 -1 开始，`right` 为所有点中最大曼哈顿距离加 1。（==定义为左开右开区间，`left`及`left`左边均满足要求，`right`及`right`右边均不满足要求==）
>    - 通过二分查找，不断调整 `left` 和 `right`，试图找到最大的边长，使得在这个边长的正方形内满足所有点的字符标签唯一的条件。
> 3. **检查条件**：
>    - 对于每一个中间值 `mid`（表示正方形的边长的一半），检查在这个边长的正方形内，是否能包含不重复字符的点。
>    - 使用一个集合来记录正方形内的字符，如果发现有重复字符则返回 `false`，否则记录字符数量并返回 `true`。
> 4. **返回结果**：
>    - 二分查找结束后，`left` 为满足条件的最大边长。通过检查这个边长，可以得到正方形内最多包含的点的数量。

```c++
class Solution {
public:
    // 检查在正方形边长为 tar 的情况下是否可以满足条件
    // 更新满足条件的点的数量到 res 中
    bool check(vector<int>& nums, int tar, string s, int &res) {
        set<char> union_set; // 使用集合来检查字符是否重复
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] <= tar) { // 如果点在正方形内
                if (union_set.count(s[i])) // 如果已经有相同字符则返回 false
                    return false;
                union_set.insert(s[i]); // 插入当前字符到集合
            }
        }
        res = union_set.size(); // 更新在正方形内的点的数量
        return true; // 如果满足条件返回 true
    }

    // 计算正方形内最多可以包含的点数
    int maxPointsInsideSquare(vector<vector<int>>& points, string s) {
        vector<int> max_nums; // 存储每个点到原点的最大曼哈顿距离
        for (int i = 0; i < points.size(); i++) {
            // 计算每个点到原点的最大曼哈顿距离
            max_nums.emplace_back(max(abs(points[i][0]), abs(points[i][1])));
        }
        
        // 定义二分查找的范围
        int left = -1, right = *max_element(max_nums.begin(), max_nums.end()) + 1;
        int mid;
        int res = 0; // 保存最终结果

        // 二分查找
        while (left + 1 < right) {
            mid = left + (right - left) / 2; // 计算中间值
            if (check(max_nums, mid, s, res)) // 检查当前中间值能否满足条件
                left = mid; // 如果满足条件，左边界右移
            else
                right = mid; // 否则右边界左移
        }
        return res; // 返回最大包含点数
    }
};
```

### [2064. 分配给商店的最多商品的最小值](https://leetcode.cn/problems/minimized-maximum-of-products-distributed-to-any-store/)

> 思路：最小化最大值，实质上还是求最小，满足`check`收缩右边界，不满足收缩左边界即可

```c++
class Solution {
public:
    //检测当前tar是否满足条件
    bool check(vector<int> &quantities,int n,int tar){
        int sum=0;
        for(int i=0;i<quantities.size();i++){
            sum+=(quantities[i]+tar-1)/tar;//向上取整.
            if(sum>n)
                return false;
        }
        return true;
    }
    int minimizedMaximum(int n, vector<int>& quantities) {
        //定义为左开右开区间，left及left左边均不满足条件，right以及right右边均满足条件.
        int left=0,right=*max_element(quantities.begin(),quantities.end());
        int mid;
        while(left+1<right){
            mid=left+(right-left)/2;
            if(check(quantities,n,mid))
                //如果当前mid满足条件，往左边收缩
                right=mid;
            else
                left=mid;
        }
        return right;
    }
};
```

### [1760. 袋子里最少数目的球](https://leetcode.cn/problems/minimum-limit-of-balls-in-a-bag/)

> 思路：查看题目，最终剩余袋子总数为`nums.size()+maxOperations`，观察发现，答案具有单调性，如果`mid+1`是每个袋子最大数量满足题目条件，那么`mid`也能满足.如果`mid`不满足条件。那么`mid+1`同样不满足。因此二分答案，采用左开右开区间。左边界设置为`0`右边界设置为`max_element`。`check`函数检查，以`mid`为最大数量，最终需要多少个袋子，如果小于最终袋子数量，满足，否则不满足。

```c++
class Solution {
public:
    // 检查当前的最大容量 tar 是否满足条件，即能否将所有的球分配到 maxBag 个袋子中
    bool check(vector<int> &nums, int tar, int maxBag) {
        int sum = 0; // 记录所需的袋子数量
        for (int i = 0; i < nums.size(); i++) {
            // 计算当前 nums[i] 个球需要多少个容量为 tar 的袋子，向上取整
            sum += (nums[i] + tar - 1) / tar;
            // 如果所需的袋子数量超过 maxBag，返回 false
            if (sum > maxBag)
                return false;
        }
        // 如果可以在 maxBag 个袋子中装下所有的球，返回 true
        return true;
    }

    int minimumSize(vector<int>& nums, int maxOperations) {
        // maxOperations + nums.size() 代表最终有多少个袋子
        int maxBag = nums.size() + maxOperations;
        // 初始化二分查找的左边界 left 和右边界 right
        int left = 0, right = *max_element(nums.begin(), nums.end());
        int mid;

        // 二分查找，寻找满足条件的最小 tar
        while (left + 1 < right) {
            mid = left + (right - left) / 2; // 计算中间值
            if (check(nums, mid, maxBag)) {
                // 如果当前 mid 满足条件，则收缩右边界
                right = mid;
            } else {
                // 否则，收缩左边界
                left = mid;
            }
        }

        // 返回满足条件的最小 tar
        return right;
    }
};
```

### [1631. 最小体力消耗路径（学习图论待优化）](https://leetcode.cn/problems/path-with-minimum-effort/)

> 思路1（自己）：采用二分+`bfs`，判断每个`tar`能够从左上角到达右下角。（性能极差，check函数不是自己写的）

```c++
class Solution {
public:
    bool check(vector<vector<int>>& heights, int tar) {
        int n = heights.size();
        int m = heights[0].size();
        vector<vector<bool>> visited(n, vector<bool>(m, false));
        queue<pair<int, int>> q;
        vector<int> directions = {-1, 0, 1, 0, -1}; // 方向数组，方便计算四个方向

        // 将起点 (0, 0) 加入队列
        q.push({0, 0});
        visited[0][0] = true;

        while (!q.empty()) {
            auto [x, y] = q.front();
            q.pop();

            // 如果到达终点 (n-1, m-1)，返回 true
            if (x == n - 1 && y == m - 1) return true;

            // 遍历四个方向
            for (int k = 0; k < 4; k++) {
                int nx = x + directions[k];
                int ny = y + directions[k + 1];

                // 判断是否越界以及是否访问过
                if (nx >= 0 && nx < n && ny >= 0 && ny < m && !visited[nx][ny]) {
                    int diff = abs(heights[nx][ny] - heights[x][y]);
                    // 如果当前高度差小于等于 tar，加入队列
                    if (diff <= tar) {
                        q.push({nx, ny});
                        visited[nx][ny] = true;
                    }
                }
            }
        }
        return false;
    }
    int minimumEffortPath(vector<vector<int>>& heights) {
        //通过二分答案
        //左开右开 left及left左边一定不满足，right及right右边一定满足。
        int left=-1,right=1e6+1;
        int mid;
        while(left+1<right){
            mid=left+(right-left)/2;
            if(check(heights,mid))
                right=mid;
            else
                left=mid;
        }
        return right;
    }
};
```

### [2439. 最小化数组中的最大值](https://leetcode.cn/problems/minimize-maximum-of-array/)

> 思路：二分答案，判断最前面一个元素经过调整，是否会大于`tar`。
>
> ### 思路总结
>
> 1. **方法**：
>    - 使用二分查找和检查函数来找到满足条件的最小最大值。（采用左开右开区间做法）
> 2. **具体步骤**：
>    1. **初始化**：
>       - `left` 设置为 `-1`。
>       - `right` 设置为数组中最大元素的值加 1（`(long long) *max_element(nums.begin(), nums.end()) + 1`）。
>    2. **二分查找**：
>       - 在 `(left, right)` 区间内进行二分查找。
>       - 计算中间值 `mid = left + (right - left) / 2`。
>       - 调用检查函数 `check(nums, mid)` 判断是否能够使所有元素都不超过 `mid`。
>    3. **检查函数 `check`**：
>       - 从数组的最后一个元素开始向前遍历。
>       - 计算累积的超出部分 `extra`，即 `max(nums[i] + extra - tar, 0LL)`。
>       - 检查第一个元素加上累积的超出部分是否不超过目标值 `tar`。
>       - 如果不超过，则返回 `true`；否则，返回 `false`。
>    4. **调整二分区间**：
>       - 如果 `check` 返回 `true`，表示当前 `mid` 满足条件，收缩右边界（`right = mid`）。
>       - 如果 `check` 返回 `false`，表示当前 `mid` 不满足条件，收缩左边界（`left = mid`）。
>    5. **返回结果**：
>       - 最终 `right` 的值即为满足条件的最小最大值。

```c++
class Solution {
public:
    // 检查是否可以将数组调整为所有元素都不超过 tar
    bool check(vector<int> &nums, long long tar) {
        long long extra = 0; // 用于存储从右侧累积的多余部分
        // 从后向前遍历数组
        for (int i = nums.size() - 1; i > 0; i--) {
            // 计算当前位置的多余部分，并将其转移到前一个元素
            extra = max(nums[i] + extra - tar, 0LL);
        }
        // 检查最前面的元素加上多余部分是否不超过 tar
        return nums[0] + extra <= tar;
    }

    int minimizeArrayValue(vector<int>& nums) {
        // 二分查找的左右边界初始化
        // left 设置为 -1，表示无效的最小值
        // right 设置为数组中最大元素的值加 1
        long long left = -1, right = (long long)*max_element(nums.begin(), nums.end()) + 1;
        long long mid;
        
        // 开始二分查找
        while (left + 1 < right) {
            // 计算中间值
            mid = left + (right - left) / 2;
            // 检查当前 mid 是否满足条件
            if (check(nums, mid))
                // 如果满足条件，收缩右边界
                right = mid;
            else
                // 如果不满足条件，收缩左边界
                left = mid;
        }
        // 返回满足条件的最小最大值
        return right;
    }
};
```

### [74. 搜索二维矩阵](https://leetcode.cn/problems/search-a-2d-matrix/)

> 思路1：将二维矩阵转一维矩阵。利用二分搜索，左开右开区间，左边界`-1`,右边界`matrix[0].size()*matrix.size()`
>
> 思路2：采用排除法，每次检查右上角元素，如果该元素大于`tar`排除比该元素大的，即`j--`,如果该元素小于`tar`排除比该元素小的.即`i++`

```c++
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int rows = matrix.size();
        int cols = matrix[0].size();
        int left = 0, right = rows * cols - 1;
        
        // 二分查找
        while (left <= right) {
            int mid = left + (right - left) / 2;
            int mid_value = matrix[mid / cols][mid % cols];
            
            if (mid_value == target) {
                return true;
            } else if (mid_value < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return false;
    }
};


//排除法
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int i=0,j=matrix[0].size()-1;
        while(i<matrix.size()&&j>=0){
            if(matrix[i][j]<target)
                //比该数小的都排除
                i++;
            else if(matrix[i][j]>target)
                //比该数大的都排除
                j--;
            else
                return true;
        }
        return false;
    }
};
```

### [:imp: 162. 寻找峰值（值得好好思考）](https://leetcode.cn/problems/find-peak-element/)

> 思路：由红蓝染色法，定义蓝色为峰值左边元素，红色为峰值或者峰值右边元素。因此当`nums[mid]<nums[mid+1]`说明当前`mid`所指元素为蓝色,移动`left=mid`  如果`nums[mid]>nums[mid+1]`说明当前元素为峰值元素或峰值右边元素。移动`right=mid`。最后返回`right`。

> [!CAUTION]
>
> 解释为什么当`nums[mid]>nums[mid+1]`一直往右一定能找到峰值：由题给条件，对于所有有效的 `i` 都有 `nums[i] != nums[i + 1]`此条件告诉我们，当前的`mid`要么就是此数组最大值，要么再其右边一定有比`nums[mid]`更大的值。因此一定能够找到峰值。就算特殊情况，一直升序。由题目条件：你可以假设 `nums[-1] = nums[n] = -∞` 最后一个元素也必然为峰值元素。



> [!IMPORTANT]
>
> 思考：由此题，我认为，红蓝染色法的关键，在于定义好了`left`以及`right`所代表的含义之后，后面移动指针时候，就是根据定义来进行移动。最后返回值时候，根据定义代表的意义来进行返回。

```c++
class Solution {
public:
    int findPeakElement(vector<int>& nums) {
        //红蓝染色法，定义蓝色为峰值或者峰值左侧，红色为峰值右侧(左闭右开区间写法)
        int left=0,right=nums.size();
        int mid;
        while(left<right){
            mid=left+(right-left)/2;
            if(mid<nums.size()-1&&nums[mid]<nums[mid+1])
                //mid所在位置为蓝色,
                left=mid+1;
            else
                right=mid;
        }
        return left;
    }
};
```

### [:apple: 153. 寻找旋转排序数组中的最小值（很好的题目）](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/)

> 思路：明确目标，我们需要找到最小值。根据红蓝染色法，我们定义蓝色为`>`最后一个数的值，而红色为`<=`最后一个数的值。
>
> 定义`left`以及`left`左边为蓝色，`right`以及`right`右边为红色。最后`left`指向位置为数组最大元素，`right`指向位置为数组最小元素。
>
> 由于最后一个元素已经满足定义将其染为红色。定义为开区间。

```c++
class Solution {
public:
    int findMin(vector<int>& nums) {
        int n=nums.size()-1;
        int left=-1,right=nums.size()-1;
        //定义为左开右开区间 left和left左边均是>= nums[len-1]  right及right右边均是<nums[len-1]
        int mid;
        while(left+1<right){
            mid=left+(right-left)/2;
            if(nums[mid]<nums[n])
                right=mid;
            else
                left=mid;
        }
        return nums[right];
    }
};
```

### [:muscle: 33. 搜索旋转排序数组（综合了前两道）](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

> 思路：此题重点是需要找到分界点，如何找到分界点呢？我们可以通过和最后一个元素做比较，利用红蓝染色法，蓝色定义为`>`最后一个元素，红色定义为`<=`最后一个元素。
>
> 通过一个开区间的二分法，`(-1,nums.size()-1)`，`nums[left]>nums[len-1]`同时`nums[right]<=nums[len-1]`。判断`nums[mid]`是否`>nums[len-1]`.如果满足，移动`left=mid`.否则移动`right=mid`.最后`left`将指向数组最大值==(定义nums[-1]=**∞**)==。`right`将指向数组最小值。
>
> 判断`target`具体属于哪一段，左段还是右段。再使用一次二分即可。
>
> ### 总结思路
>
> 1. **目标**：
>    - 在旋转排序数组中找到给定目标值 `target` 的索引。
> 2. **步骤**：
>    1. **寻找旋转点**：
>       - 使用二分查找法，通过比较中间值 `nums[mid]` 和数组最后一个元素 `nums[n]` 来确定旋转点。
>       - 旋转点是数组中从左段到右段的分界点。通过二分查找，可以确定 `left` 指向最大值，`right` 指向最小值。
>    2. **确定目标值所在的段**：
>       - 根据 `target` 与 `nums[n]` 的比较，确定 `target` 是在左段还是右段。
>       - 如果 `target` 大于 `nums[n]`，说明 `target` 在左段，重置 `left` 为 `-1`。
>       - 否则，说明 `target` 在右段，重置 `right` 为 `nums.size()`。
>    3. **二分查找目标值**：
>       - 使用 `lower_bound` 函数在确定的段中进行二分查找，找到第一个不小于 `target` 的元素位置。
>       - 返回找到的位置，或者返回 `-1` 表示未找到。

```c++
class Solution {
public:
    // lower_bound 函数，找到第一个不小于 target 的元素位置
    // 采用开区间写法：nums[left] < target, nums[right] >= target
    int lower_bound(vector<int> &nums, int target, int left, int right) {
        int mid;
        while (left + 1 < right) { // 当 left + 1 < right 时继续搜索
            mid = left + (right - left) / 2; // 计算中间索引，避免直接加法可能导致的溢出
            if (nums[mid] >= target) {
                right = mid; // 如果 nums[mid] >= target，右边界左移
            } else {
                left = mid; // 如果 nums[mid] < target，左边界右移
            }
        }
        return nums[right] == target ? right : -1; // 返回第一个不小于 target 的元素位置，或者 -1 表示未找到
    }

    // 主函数，查找旋转排序数组中的目标值
    int search(vector<int>& nums, int target) {
        int left = -1, right = nums.size() - 1; // 初始化左边界为 -1，右边界为数组的最后一个索引
        int n = nums.size() - 1;
        int mid;

        // 找到旋转点，left 指向最大值，right 指向最小值
        while (left + 1 < right) {
            mid = left + (right - left) / 2;
            if (nums[mid] > nums[n]) {
                left = mid; // mid 位于左段，向右移动 left
            } else {
                right = mid; // mid 位于右段，向左移动 right
            }
        }
        // 此时 left 指向最大值，right 指向最小值

        // 判断 target 在左段还是右段
        if (target > nums[n]) {
            left = -1; // target 在左段
        } else {
            right = nums.size(); // target 在右段
        }

        // 在相应段中进行二分查找
        return lower_bound(nums, target, left, right);
    }
};

```

### [3281. 范围内整数的最大得分](https://leetcode.cn/problems/maximize-score-of-numbers-in-ranges/)

> 思路：由于需要在每个区间，取值，因此我们将区间的左端点进行排序。再观察答案具有二义性。设从第一个区间选了数字 x，那么第二个区间所选的数字至少为 `x+score`，否则不满足得分的定义。由于得分越大，所选数字越可能不在区间内，有单调性，可以二分答案。我们通过二分进行枚举答案
>
> `check`函数：判断当前选择的答案是否符合要求，我们相邻两个区间的选择点的距离**至少**要为`score`。因此我们跟区间的左端点进行对比，选择的点$x_i=max(x_{i-1}+score,start[i])$。同时必须要满足选择的点小于右端点。即 $x_i\le start[i]+d$
>
> 关键点：
>
> 1. 定义l及其左边满足要求 即 `check()==true`时移动l=mid  ，r及其右边不满足要求，即 `check()==false`时移动r=mid  最后返回l即可。

```c++
class Solution {
public:
    bool check(vector<int> &nums, int d, int x) {
        long long begin = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            if(begin+x<nums[i]){
                begin=nums[i];
                continue;
            }
            if (begin + x > nums[i] + d) return false;
            begin = begin + x;
        }
        return true;
    }
    
    int maxPossibleScore(vector<int>& start, int d) {
        sort(start.begin(), start.end());

        int l = -1, r = 2e9+10, mid;
        while (l + 1 < r) {
            mid = l + (r - l) / 2;
            if (check(start, d, mid))
                l = mid;
            else
                r = mid;
        }
        return l;
    }
};

```

### [2517. 礼盒的最大甜蜜度](https://leetcode.cn/problems/maximum-tastiness-of-candy-basket/)

> 思路：与上题类似，如果题目中有「最大化最小值」或者「最小化最大值」，一般都是二分答案。本题的答案具有二义性，对于本题来说,甜蜜度越大，能够选择的糖果就越小，反之则越多。因此可以使用二分。定义$f(d)$表示为甜蜜度至少为$d$时最多能够选择多少糖果。有：
>
> * 如果$f(d)\ge k$，说明答案至少为d
> * 如果$f(d) < k$​，说明答案最多是d-1
>
> 关键点：
>
> 1. `check`函数：我们需要从小到大进行贪心选择，从 *price*[0] 开始选；假设上一个选的数是 *pre*，那么当 *price*[*i*]≥*pre*+*d* 时，才可以选 *price*[*i*]。

```c++
class Solution {
public:
    bool check(vector<int> &nums,int k,int x){
        int count=1;
        int begin=nums[0];
        for(int i=1;i<nums.size();i++){
            if(nums[i]-begin>=x){
                count++;
                begin=nums[i];
                 if(count==k)return true;
            }
           
        }
        return false;
    }
    int maximumTastiness(vector<int>& price, int k) {
        
        sort(price.begin(),price.end());
        int l=-1,r=1e9+10,mid;
        while(l+1<r){
            mid=l+(r-l)/2;
            if(check(price,k,mid))
                l=mid;
            else
                r=mid;
        }
        return l;
    }
};
```

