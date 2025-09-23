# 1. 两数之和
给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target 的那 两个 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案，并且你不能使用两次相同的元素。

你可以按任意顺序返回答案。

```
class Solution
{
public:
    vector<int>twoSum(vector<int>nums,int target)
    {   unordered_map<int,int>mp;
        for(int i=0;i<nums.size();++i)
        {
            if(mp.find(target-nums[i])!=mp.end()){return{mp[target-nums[i]],i};}
            else{mp[nums[i]]=i;}
        }
    return {};
    }
};
```

1.看到找东西，想到unordered_map做。

2.map有find和count两种方法，find快。

3.{}表示空Vector

---

# 49. 字母异位词分组

给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。
```
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        vector<vector<string>>ans;
        unordered_map<string,vector<string>> mp;
        for(string str:strs)
        {   string temp = str;
            sort(temp.begin(),temp.end());
            mp[temp].push_back(str);
        }
        for(auto it:mp)
        {
            ans.push_back(it.second);
        }
    return ans;
    }
};
```
1. 异位词的特性就是sort后相同，归纳到一类的话用unordered_map.
2. unordered_map遍历用 for auto v:mp ,取value 用 v.second, 不是v.second()

---

# 128. 最长连续序列

给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 O(n) 的算法解决此问题。

```
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> s(nums.begin(),nums.end());
        int ans=0;
        for(auto i:s)
        {
            if(s.count(i-1)){continue;}
            else{
                int cnt = 0; int curr = i;
                while(s.count(curr++)){cnt++;ans = max(ans,cnt);}
            }
        }
    return ans;
    }
};
```

1. 如果数字连续，判断是不是开头。

2. 用set去重并且o(1)找到。

---

# 83.移动零
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。

请注意 ，必须在不复制数组的情况下原地对数组进行操作。

```
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int ins = 0;
        for(int i=0;i<nums.size();++i)
        {
            if(nums[i]!=0)
            {
                swap(nums[ins++],nums[i]);
            }
        }
    }
};
```
1.不是零的放前面，零自动放后面了，同时保证顺序。

---

# 11. 盛最多水的容器

给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。

找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

说明：你不能倾斜容器。

```
class Solution {
public:
    int maxArea(vector<int>& height) {
        int l = 0,r = height.size()-1; int ans = 0;
        while(l<r)
        {
            ans = max(ans, min(height[r],height[l])*(r-l));
            if(height[l]>height[r])r--;
            else {l++;}
        }
    return ans;
}
};
```
1.循环中不要同时进入两个if

---

# 15.三数之和
给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请你返回所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组。

```
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> ans;
        sort(nums.begin(), nums.end());
        int n = nums.size();

        for (int l = 0; l < n - 2; ++l) {
            if (l > 0 && nums[l] == nums[l - 1]) continue; // 去重 l
            if (nums[l] > 0) break;                         // 剪枝

            int mid = l + 1, r = n - 1;
            int target = -nums[l];
            while (mid < r) {
                int s = nums[mid] + nums[r];
                if (s == target) {
                    ans.push_back({nums[l], nums[mid], nums[r]});
                    ++mid; --r;
                    while (mid < r && nums[mid] == nums[mid - 1]) ++mid; // 去重 mid
                    while (mid < r && nums[r] == nums[r + 1]) --r;       // 去重 r
                } else if (s < target) {
                    ++mid;
                } else {
                    --r;
                }
            }
        }
        return ans;
    }
};

```

1.用for确定left，while在mid和right中迭代

2.锁定一组后也要更新mid 和 right

3.外层去重：在进入双指针之前就做：if (l>0 && nums[l]==nums[l-1]) continue;

4.内层去重：这一步是在“锁定一组解”后去掉与这组解数值完全相同的其它组合，防止重复输出。

---

# 42.接雨水
给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

```
class Solution {
public:
    int trap(vector<int>& height) {
        
        vector<int> left_max(height.size(),0);
        vector<int> right_max(height.size(),0);
        int ans = 0;

        int l_max = 0,r_max = 0;

        for(int i =0;i<height.size();++i)
        {   
            l_max=max(l_max,height[i]);
            left_max[i]=l_max;
        }

        for(int j =height.size()-1;j>=0;--j)
        {   
            r_max=max(r_max,height[j]);
            right_max[j]=r_max;
        }

        for(int i =0;i<height.size();++i)
        {   
            ans+= min(left_max[i],right_max[i])-height[i];

        }
    return ans;
    }
};
```

1.vector<int>v (nums,a)表示初始化nums个a

2.当前能盛水高度 = 左右最大值中的最小的一个-当前高度 ->从左遍历 从右遍历

---

# 3. 无重复字符的最长子串
给定一个字符串 s ，请你找出其中不含有重复字符的 最长 子串 的长度。

```
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int n = s.size();if(n==0)return 0 ;
        int ans = 0;
        unordered_map<int,int> mp;
        int l = 0;
        for(int r =0;r<n;++r)
        {
            auto it= mp.find(s[r]-'a');
            if(it!= mp.end() && it->second>=l) {l=it->second+1;}

            mp[s[r]-'a']=r;
            ans =max(ans,r-l+1);
        }
    return ans;
}
};
```

1.用l表示边界，r表示目前看的字符

2.新增值在窗内就更新l，否则就更新r

3.每次都计算窗是不是最大？

---
# 438. 找到字符串中所有字母异位词
给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

```
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        int s_len = s.size(),p_len = p.size();
        vector<int> ans;
        vector<int>mp(26,0),cor(26,0);
        for(int l=0;l<p_len;++l)
        {
            cor[p[l]-'a']++;
        }

        for(int r=0;r<s_len;++r)
        {   if(r<p_len)
            mp[s[r]-'a']++;
            else
            {
                mp[s[r-p_len]-'a']--;
                mp[s[r]-'a']++;
            }
            if(cor == mp)
            {ans.push_back(r-p_len+1);}  
        }   
    return ans;
}
    
};
```
1. 用 数量来做，unordered_map<int,int> 比较 mp == cor，但窗口滑出时没有把计数为 0 的键删掉，if (--win[key] == 0) win.erase(key);

2. 要么用两个vector计数，初始化26个'0'


---

# 560. 和为 K 的子数组
给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的子数组的个数 。

子数组是数组中元素的连续非空序列。

```
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<long long,int>mp;
        long long sum=0;
        int ans = 0;
        mp[0] = 1;
        for(int i=0;i<nums.size();++i)
        {
            sum = sum + nums[i];

            if(mp.count(sum-k)){ans+=mp[sum-k];}
            mp[sum]++;

        }
    return ans;
}
};
```

1.看到计算子数组和，就是计算0～末尾 - 0～起始

2.用map的sum做key，**和为sum的次数/数量** 作value : unordered_map<long long, int> cnt; // 前缀和 -> 出现次数

---

# *239. 滑动窗口最大值

给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

返回 滑动窗口中的最大值 。

```
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int n = nums.size();
        if (n == 0 || k == 0) return {};

        deque<int>dq;vector<int>ans;
        for(int i=0;i<nums.size();++i)
        {   
            while(!dq.empty() && dq.front()<=i-k){dq.pop_front();}

            while(!dq.empty() && nums[dq.back()]<=nums[i]){dq.pop_back();}    

            dq.emplace_back(i);

            if(i>=k-1)ans.push_back(nums[dq.front()]);
        }
    return ans;  
    }
};
```
**记住：**

**掐头去尾：** 掐老头，去小尾。

1.看到滑动窗口最大值，用双端队列deque，**保存下标**。并保持**对应的值**从队头到队尾**递减**。

2.维护递减性：当 dq 尾部对应的值 <= nums[i]，从队尾弹出，直到队列递减。

3.把当前下标 i 压入队尾。

4.当形成完整窗口（i >= k-1）时，答案加入 nums[dq.front()]（此时队头就是窗口最大值）。

---

# 76. 最小覆盖子串

给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。

```
！！！！
```
---

# 53. 最大子数组和
给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

子数组是数组中的一个连续部分。

```
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int curr = 0;
        int ans = -INT_MAX;
        for(int i = 0;i<nums.size();++i)
        {
            curr = max(curr+nums[i],nums[i]);
            ans = max(curr,ans);
        }
    return ans;
}
};
```

1.明显用dp，当前表示子数组的末尾。

2.最大值为INT_MAX;

---

# 56. 合并区间

以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。

```
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        vector<vector<int>> ans;
        sort(intervals.begin(),intervals.end());
        vector<int>curr; 
        for(auto interval:intervals)
        {
            if(curr.empty()) curr = {interval[0],interval[1]};
            
            else if(interval[0]>=curr[0] && interval[0]<=curr[1]){curr[1]=max(curr[1],interval[1]);}
            else {ans.push_back(curr);curr =interval;}
        }
        ans.push_back(curr);
    return ans;
}
};
```
1. vector<vector<int>> 的排序是按照每行第一个值排序的（集合start）；

2. empty时需要放一个curr，且别忘最后需要将curr加入。 

3.用max简化逻辑

---

# 189. 轮转数组

给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。

```
class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        int n = k % nums.size();
        reverse(nums.begin(),nums.end());
        reverse(nums.begin(),nums.begin()+n);
        reverse(nums.begin()+n,nums.end());
    }
};
```

1. 翻转三次。左闭右开。 

2. 先取余！！

---

# 238. 除自身以外数组的乘积

给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。

题目数据 保证 数组 nums之中任意元素的全部前缀元素和后缀的乘积都在  32 位 整数范围内。

请 不要使用除法，且在 O(n) 时间复杂度内完成此题

```
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        vector<int>pre(nums.size(),1);
        vector<int>pos(nums.size(),1);
        vector<int>ans(nums.size(),1);
        for(int i =1;i<nums.size();i++)
        {
            pre[i]=pre[i-1]*nums[i-1];
        }
        for(int j=nums.size()-2; j>=0 ; --j)
        {
            pos[j]=pos[j+1]*nums[j+1];
        }

        for(int i =0;i<nums.size();i++)
        {
            ans[i]=pre[i]*pos[i];
        }
    return ans;
}
};
```

1. 前缀积 * 后缀积

2.注意前后缀是i-1和j+1


---

# 41. 缺失的第一个正数

给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。

请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。

---

# 73. 矩阵置零

给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。

```
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        bool firstr =false,firstc= false;
        int m= matrix.size(),n=matrix[0].size();
        for(int i = 0;i<m;++i) {if (matrix[i][0]==0){firstc =true;break;}}
        for(int j = 0;j<n;++j) {if (matrix[0][j]==0){firstr =true;break;}} 
        
        for(int i = 1;i<m;++i)
        {
            for(int j = 1;j<n;++j)
                {
                    if (matrix[i][j]==0)
                    {
                        matrix[i][0]=0;
                        matrix[0][j]=0;

                    }
                }
        }

        for(int i = 1;i<m;++i){
            if(matrix[i][0]==0)
            {
                for(int j = 0;j<n;++j)
                {matrix[i][j]=0;}
            }
        }
        for(int j = 1;j<n;++j){
            if(matrix[0][j]==0)
            {
                for(int i = 0;i<m;++i)
                {matrix[i][j]=0;}
            }
        }
        if (firstc){
            for(int i = 0;i<m;++i)
             {matrix[i][0]=0;}
        }
       
       if (firstr){
            for(int j = 0;j<n;++j)
             {matrix[0][j]=0;}
        }

    }
};
```

1.判断第一行/第一列是否有0，保存为两个值

2.判断每一行/每一列是否有0，如果有就将行首/列首元素置为0

---

# 54. 螺旋矩阵

给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。

```
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        vector<int>ans;
        int l=0,r=n-1,t=0,b=m-1;

        while(l<=r && t<=b)
        {
            for(int i = l; i<=r; ++i){ans.push_back(matrix[t][i]);}t++;

            for(int i = t; i<=b; ++i){ans.push_back(matrix[i][r]);}r--;
        
            if(l<=r && t<=b){
                for(int i = r; i>=l ; --i){ans.push_back(matrix[b][i]);}
            }b--;
            if(l<=r && t<=b){
                for(int i = b; i>=t ; --i){ans.push_back(matrix[i][l]);}
            }l++;
        }
    return ans;
    }
};
```
easy题目

1.第三段、第四段前，因为收缩过，所以要重新判断满足条件。

2. ++操作在括号外；

---

# 48.旋转图像
给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。

你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。

```
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        for(int i =0;i<matrix.size()-1;++i)
        {
            for(int j= i+1; j<matrix.size(); ++j)
            {
                swap(matrix[i][j],matrix[j][i]);
            }
        }

        for(int i =0;i<matrix.size();++i)
        {
            for(int j= 0; j<matrix.size()/2; ++j)
            {
                swap( matrix[i][j] , matrix[i][matrix.size()-1-j] );
            }
        }
    }
};
```
1.旋转 = 转置 + 左右对称

---
# 240. 搜索二维矩阵 II
编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：

每行的元素从左到右升序排列。
每列的元素从上到下升序排列。

```
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size();
        int n = matrix[0].size();
        int i = m-1;
        int j = 0;
        while (i>=0&& j <n)
        {   if(matrix[i][j]==target){return true;}
            else if(matrix[i][j]<target){j++;}
            else {i--;}
        }
    return false;
    }
};
```
easy

1.从左下开始，大于当前，排除本行，小于当前，排除本列；

---

# 160. 相交链表

给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。

```
struct ListNode {
      int val;
      ListNode *next;
      ListNode(int x) : val(x), next(NULL) {}
  };
 
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode *A = headA, *B = headB;
        while(A!=B)
        {
            A= (A==nullptr? headB: A->next);
            B= (B==nullptr? headA: B->next);
        }
    return A;
    }
};
```
1. l1走完走l2,l2走完走l1，距离相同

2. 结束条件为A！=B ，那么判断时就是同为nullptr，变换到对方的头。

3. 不要忘记新建节点。

---

# 206. 反转链表

给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。

```
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        unordered_map<ListNode*,ListNode*> mp;
        ListNode* a = head;
        ListNode* prev = nullptr;
        while(a)
        {   ListNode* b;
            b=a->next;
            a->next= prev;
            prev = a;
            a=b;
        }
        return prev;
    }
};
```
1.按照定义来做：遍历+接上已有链表

---
# 234. 回文链表

给你一个单链表的头节点 head ，请你判断该链表是否为回文链表。如果是，返回 true；否则，返回 false 。

```
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        unordered_map<ListNode*,ListNode*> mp;
        ListNode* a = head;
        ListNode* prev = nullptr;
        while(a)
        {   ListNode* b;
            b=a->next;
            a->next= prev;
            prev = a;
            a=b;
        }
        return prev;
    }
    bool isPalindrome(ListNode* head) {
        ListNode* a = head;
        ListNode* b = head;
        while(b!=nullptr && b->next!=nullptr)
        {
            b=b->next->next;
            a=a->next;
        }
        ListNode* second  =reverseList(a);

        ListNode* p1 = head;
        ListNode* p2 = second;

        while(p2)
        {
            if(p1->val==p2->val)
            {
                p1=p1->next;
                p2=p2->next;
            }
            else
            {
                return false;
            }
        }
    return true;
    }
};
```
1.回文->前后相同->找到中点->快慢指针

2.只要while(翻转)并与整个链比较就够了。

---

# 141. 环形链表

给你一个链表的头节点 head ，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递 。仅仅是为了标识链表的实际情况

```
class Solution {
public:
    bool hasCycle(ListNode *head) {
        ListNode* a = head;
        ListNode* b = head;
        while(b && b->next)
        {
            b=b->next->next;
            a=a->next;
            if(a==b){return true;}
        }
        return false;
    }
};
```
1.快的走到底了，那就没有环

2.碰头了就是有环

---

# 142. 环形链表 II

给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。

```
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode* a = head;
        ListNode* b = head;
        while(b && b->next)
        {
            b=b->next->next;
            a=a->next;
            if(a==b){
                ListNode* slow = head;
                while(slow!=b){slow=slow->next;b=b->next;}
                    return b;
                }
        }
        return nullptr;
    }
    

};
```

1.保留快慢指针，如果a==b那就是有环，就把慢的放在头，再稳步走，必能在入口碰头。

2.推导

链表从头到环入口的长度：x

环长：c

从入口到相遇点沿环前进的距离：y

慢指针速度 1，快指针速度 2。第一次相遇时，慢指针总步数记为 t。

因为慢指针从头走到相遇点，必然先走 x 到达入口，再在环内走 y 到相遇点，所以：

t = x + y                             …(1)

快指针走了 2t 步。它也要走 x 才能进环，然后在环里比慢指针多绕了若干圈再到相遇点。设多绕了 k 圈（k≥1），则：

2t = x + y + k*c                      …(2)


用 (2) − (1) 得到：

t = k*c  ⇒  x + y = k*c  ⇒  x = k*c - y      …(3)

---

# 21. 合并两个有序链表

将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

```
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode * a = list1;
        ListNode * b = list2;
        ListNode dummy(0);
        ListNode *curr=&dummy;
        while(a && b)
        {
            if(a->val <= b->val){curr->next= a; a=a->next;}
            else {curr->next = b;b=b->next;}
            curr= curr->next;
        }
        if(a){curr->next= a;}
        if(b){curr->next= b;}
        return dummy.next;

    }
};
```
1.哑节点的构造为：
ListNode dummy(0);ListNode *curr=&dummy;

使用为：dummy.next;
        
2.if(a){curr->next= a;} 直接加上就行

---
# 2. 两数相加

给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

```
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode dummy(0);            // 哑头结点
        ListNode* head = &dummy;
        ListNode* prev= head;
        int num=0;int cnt=0;

        while(l1 || l2|| cnt)
        {   int x,y;
            x = l1 ? l1->val : 0;
            y = l2 ? l2->val : 0;
            num = (x+y+cnt)%10;
            cnt = (x+y+cnt)/10;

            ListNode * curr = new ListNode(num);
            
            prev->next= curr;
            prev =prev->next;
            if(l1)l1=l1->next;
            if(l2)l2=l2->next;
        }
       
        if(cnt)
        {
            
            prev->next=new ListNode(cnt);
        }
    return head->next;
}
};
```
1.语法：ListNode * curr = new ListNode(num);或者ListNode dummy(0);ListNode *curr=&dummy;初始化

2. 合并 l1,l2,carry三种情况，用 x = l1 ? l1->val : 0; 做合并

3.不要忘记最后一次carry的判定

---
# 19. 删除链表的倒数第 N 个结点

给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

```
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode dummy(0);
        ListNode * a = &dummy;dummy.next = head;
        ListNode *f =a,*curr= a;

        for(int i =0;i<n;++i)
        {
            curr = curr->next;
        }
        while(curr->next)
        {
            curr = curr->next;
            f=f->next;
        }
        f->next=f->next->next;
        return a->next;

    }
};
```
1.fast先走n步，再和slow同步走，fast走到最后，slow就是倒数k个结点。

2.但是去掉k，必须到k-1，那就前面加上dummy

3.dummy 不要忘记和head链接

---

# 24. 两两交换链表中的节点

给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。

```
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        ListNode * dummy = new ListNode(0);
        dummy->next = head;
        ListNode * a =dummy;
        
        while(a->next && a->next->next)
        {   ListNode* first = a->next, *second =first->next;
            first->next = second->next;
            a->next = second;
            second->next = first;
            a=a->next->next;
        }
    return dummy->next;
    }
};
```
1. 用dummy来间接表示第一个和第二个；不要搞复杂了，直接用first和second表示。

2. 换完不要忘记推进a


---
# 25. K 个一组翻转链表

给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。

k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

```

```
---

# 94 二叉树的中序遍历
给定一个二叉树的根节点 root ，返回 它的 中序 遍历 。

```
class Solution {
public:
    void dfs(TreeNode* root,vector<int>& ans)
    {
        if(!root)return;
        else{
            if(root->left){dfs(root->left,ans);}
            ans.push_back(root->val);
            if(root->right){dfs(root->right,ans);}
        }
    }
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> ans;
        dfs(root,ans);
        
    return ans;
    }
};

```

1.递归+返回累加列表，要用辅助函数。

2.二叉树遍历用辅助函数 dfs。中序遍历用 ‘前-打印-右’ 先递归左子树，再记录当前节点值，再递归右子树。

3.如果是前序遍历，就把dfs改变顺序

---

# 104 二叉树的最大深度
给定一个二叉树 root ，返回其最大深度。

二叉树的 最大深度 是指从根节点到最远叶子节点的最长路径上的节点数。

```
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(!root){return 0;}
        return 1+max(maxDepth(root->left),maxDepth(root->right) );
    }
};
```

1.递归，无累加列表，不用辅助函数

2.用函数定义做：最大深度 = 当前层（1）+ 子层（左/右）的最大深度。


---

# 226. 翻转二叉树
给你一棵二叉树的根节点 root ，翻转这棵二叉树，并返回其根节点。

```
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (!root)return nullptr;
        swap(root->left,root->right);
        invertTree(root->left);
        invertTree(root->right);
        return root;
    }
};
```

1.按照定义做：翻转本节点 = swap子节点 + 翻转两个子节点

---

# 101.对称二叉树

给你一个二叉树的根节点 root ， 检查它是否轴对称。
```
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        if(!root) return true;
        
        return isMirror(root->left,root->right)
    }
    bool isMirror(TreeNode* a,TreeNode* b)
    {   if(!a && !b ) return true;
        if(!a || !b) return false;
        if(a->val!=b->val) return false;
        else return isMirror(a->left,b->right)&& isMirror(a->right,b->left)
    }
};
```
1.注意轴对称：求堆成->左右子节点镜像->需要辅助函数-> 辅助函数（a,b）: a节点的val==b节点的val->找到下两组镜像的节点：a->left, b->right ? a->right,b->left?

---

# 543
给你一棵二叉树的根节点，返回该树的 直径 。

二叉树的 直径 是指树中任意两个节点之间最长路径的 长度 。这条路径可能经过也可能不经过根节点 root 。

两节点之间路径的 长度 由它们之间边数表示。

```



```

---

# 200. 岛屿数量

给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

```
class Solution {
public:

    void dfs(int i,int j,vector<vector<char>>& grid,vector<vector<int>>& occ)
    {
        int m = grid.size();
        int n = grid[0].size();
        if (i < 0 || i >= m || j < 0 || j >= n || occ[i][j] == 1 || grid[i][j] == '0') {
            return;
        }
        occ[i][j] = 1;

        dfs(i,j+1,grid,occ);
        dfs(i+1,j,grid,occ);
        dfs(i,j-1,grid,occ);
        dfs(i-1,j,grid,occ);
   
    }

    int numIslands(vector<vector<char>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        int cnt = 0;

        vector<vector<int>> occ(m,vector<int>(n,0));
        for(int i = 0;i<m;++i)
        {
            for(int j =0;j<n;++j)
            {
                if(grid[i][j]=='1' && occ[i][j]==0){
                    cnt++; 
                    dfs(i,j,grid,occ);
                }
            }
        }
    return cnt;
    }
};
```

1.逻辑：用一个同大小的表，表示有没有看过这一格。

2.用dfs来填写这个表：如果边界外/原表为0/已经看过，就return回到上个dfs函数中。

3.注意用return返回上层

break	跳出当前循环或switch语句	用于循环(for/while)或switch内部

return	结束当前函数执行并返回值	用于if等函数内部，结束整个函数

---

# 994. 腐烂的橘子

在给定的 m x n 网格 grid 中，每个单元格可以有以下三个值之一：

值 0 代表空单元格；

值 1 代表新鲜橘子；

值 2 代表腐烂的橘子。

每分钟，腐烂的橘子 周围 4 个方向上相邻 的新鲜橘子都会腐烂。

返回 直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 -1 。

```
class Solution {
public:
    int orangesRotting(vector<vector<int>>& grid) {
        int m =grid.size();
        int n = grid[0].size();
        int fresh = 0, cnt = -1;
        queue<pair<int,int>> rot;
        vector<pair<int,int>> directions= {{0,1},{0,-1},{1,0},{-1,0}};
        for(int i =0;i<m;++i)
        {
            for(int j =0;j<n;++j)
            {
                if(grid[i][j]==2) rot.push({i,j});
                if(grid[i][j]==1) fresh++;
            }
        }

        if (fresh == 0) return 0;
        
        while (!rot.empty())
        {   bool infected = false; 
            int size = rot.size();
            for(int i =0;i<size;++i)
            {
                auto pos = rot.front();
                rot.pop();
                for(auto direction:directions)
                    {
                        int x = pos.first + direction.first;
                        int y = pos.second + direction.second;
                        if(x>=0 && x<m && y>=0 && y<n && grid[x][y]==1)
                        {
                            grid[x][y]=2;
                            fresh--;
                            rot.push({x,y});
                        }
                    }
            }
            cnt++;
        }
        if (fresh!=0) return -1;
        else {return cnt;}

    }
};
```

1.多源BFS初始化：将所有初始腐烂的橘子加入队列 使用queue，用front(),pop(),push()

2.统计新鲜橘子：记录初始新鲜橘子的数量

3.分层BFS：每分钟处理当前层的所有腐烂橘子，感染周围的新鲜橘子（把queue中已有的记录个数遍历这么多个）

4.时间统计：每完成一层BFS，时间增加1分钟

5. 结果检查：如果最后还有新鲜橘子剩余，返回-1；否则返回总时间

6. queue和deque

---


# 207.课程表


你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。

在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。

例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。

```
class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<int>indeg(numCourses,0);vector<vector<int>>g(numCourses);
        
       
        for(int i = 0;i<prerequisites.size();++i)
        {   g[prerequisites[i][1]].push_back(prerequisites[i][0]);
            indeg[prerequisites[i][0]]++;
        }

        queue<int>q;
        for(int i = 0;i<numCourses;++i)
        {   if(indeg[i]==0) {q.push(i);} 

        }

        int cnt=0;

        while(!q.empty())
        {
            int len = q.size();
            for(int i = 0;i<len;++i)
            {   
                int temp = q.front();
                q.pop();cnt++;
                for(auto item:g[temp])
                {
                  if(--indeg[item]==0){q.push(item);}
                }
            }
            
        }

        return cnt==numCourses;
    }
};
```
1.queue不能随机存储

2.逻辑：

先把所有入度为 0 的点入队 q（所有无依赖的），循环取队头 u。认为 u 已“学完”（或已输出），**出队后计数 cnt++**；遍历 u 的所有下家 v ∈ g[u]，执行 --indeg[v]；

若某个下家 v 的入度因此变成 0，就把 v 入队。队列空了后，若 cnt == n，说明能把所有点处理完（无环、可修完）；否则有环（至少一些点入度始终 >0）。

---

# 46. 全排列

给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
```
class Solution {
public:
    void dfs(vector<int>& nums,vector<int>& used,vector<int>& path,vector<vector<int>>& ans){
        if(path.size()==nums.size()){ans.push_back(path);return;}
        for(int i =0;i<nums.size();i++)
        {   
            if (used[i]==0){
                used[i]=1;
                path.push_back(nums[i]);
                dfs(nums,used,path,ans);
                path.pop_back();
                used[i]=0;
            }
        }

    }
    vector<vector<int>> permute(vector<int>& nums) {
        int n= nums.size();
        vector<int> used(n,0);vector<int> path;vector<vector<int>> ans;
        dfs(nums,used,path,ans);
        return ans;
    }
};

```
多做

1.用dfs做，把没有用过的一直加入path，直到满。

2.used来判断用没用过，如果用过了used就置1，用了就继续dfs，dfs结束了，就是满了，就弹出末尾，然后used置0，后续换到其他位置能够继续使用。

---

# 78. 子集

给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。

解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。

```
class Solution {
public:
    void dfs(int start,vector<int>& nums,vector<int>& path,vector<vector<int>>& ans){
        ans.push_back(path);
        for(int i = start;i<nums.size();++i)
        {
            path.push_back(nums[i]);
            dfs(i+1,nums,path,ans);
            path.pop_back();
        }
    }

    vector<vector<int>> subsets(vector<int>& nums) {
        int start;vector<int>path;vector<vector<int>> ans;
        dfs(0,nums,path,ans);
    return ans;
}
};
```
多做

1. for内的循环条件不同， 使用start作为 参数，控制下一层只能从“后面”继续选

**核心对比**

**目标性质**

子集：不看顺序，同一元素集合 {1,2} 与 {2,1} 是同一个结果 → 需要避免换序产生重复。

全排列：看顺序，[1,2,3] 与 [2,1,3] 是不同结果。

**搜索树形态**

子集：深度 0..n；每到一个状态就收集一次（因为任何长度都算子集）。

全排列：深度固定为 n；只在长度 == n 时收集（必须放满）。

**去重方式 / 状态控制**

子集：用 start 控制下一层只能从“后面”继续选，避免同一层选到以前的元素而产生换序重复；不需要 used。

全排列：每层都能从所有未用元素里选，需要 used[]（或“原地交换法”的 pos）来防止重复使用同一元素。

**收集时机**

子集：进入 dfs 的第一行就 ans.push_back(path)。

全排列：当 path.size()==n 时才 ans.push_back(path)。

---

# 17. 电话号码的字母组合

给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

```
class Solution {
public:
    void dfs(int& start,const string& digits,const vector<string> & map,string& path,vector<string>& ans)
    {   
        if(start==digit.size())
        {
            ans.push_back(path);return;
        }
        int d = digits[start]-'0';
        for(char ch: map[d])
            {
                path.push_back(ch);
                dfs(start + 1,digits,map,path,ans);
                path.pop_back();
            }
        
    }
    vector<string> letterCombinations(string digits) {
        vector<string>ans;
        string path;
        vector<string> map = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        dfs(0,digits,map,path,ans);
    }
};
```
多做

1. 用start表示当前位，所有参数融进变量中。

2. vector<string> map = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};来简约表示
   
---

# 39. 组合总和

给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。

candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 

对于给定的输入，保证和为 target 的不同组合数少于 150 个。

```
class Solution {
public:
    void dfs(int start,vector<int>& path,vector<int>& candidates, int target,vector<vector<int>>& ans,int& sum){
        if(sum==target){ans.push_back(path);return;}
        
        for(int i =start;i<candidates.size();i++)
        {   
            if(sum>target){return;}
            path.push_back(candidates[i]);
            sum+=candidates[i];
            dfs(i,path,candidates,target,ans,sum);
            sum-=candidates[i];
            path.pop_back();
        }
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> ans;vector<int>path;int sum=0;
        dfs(0,path,candidates,target,ans,sum);
    return ans;
}
};
```
多写

1.随便写出来的，dfs(i,path,candidates,target,ans,sum);注意这里是i不是i+1

---

# 22. 括号生成

数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。

```
class Solution {
public:
    void dfs(int start,string & path,vector<string>& ans,int& n,int open,int close)
    {
        if (start==2*n){
            if(open==n){ans.push_back(path);return;}
            else {return;}

        }
            
        path.push_back('(');open++;
        dfs(start+1,path,ans,n,open,close);
        path.pop_back();open--;
        
        if(close<open)
        {
            path.push_back(')');
            close++;
            dfs(start+1,path,ans,n,open,close);
            path.pop_back();
            close--;
        }

    }
    vector<string> generateParenthesis(int n) {
        int num=0; string path; vector<string>ans;int open=0;int close = 0;
        dfs(0,path,ans,n,open,close);
    return ans;
    }

};
```

1.不同地方在于：右括号的判定需要加上。

2.start==2*n不是n

# 79. 单词搜索

给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用

```
class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        
    }
};
```

---
# 35. 搜索插入位置

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

请必须使用时间复杂度为 O(log n) 的算法。

```
class Solution {
public:
    int find(vector<int>& nums, int target ,int l,int r)
    {   if(l>r){return l;}
    
        int mid = l+(r-l)/2;
        if (nums[mid]==target)
            return mid;
        else if(nums[mid]>target){
            return find(nums,target,l,mid-1);
        }
        else{
            return find(nums,target,mid+1,r);
        }
    }
    int searchInsert(vector<int>& nums, int target) {
        int l=0; int r =nums.size()-1;
        return find(nums,target,l,r);
    }
};
```

1.不要忘记mid+1

2.终止条件 if(l>r){return l;}

---

# 34. 在排序数组中查找元素的第一个和最后一个位置

给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 target，返回 [-1, -1]。

你必须设计并实现时间复杂度为 O(log n) 的算法解决此问题。

```
class Solution {
public:
    int findFirst(vector<int>& nums, int target) {
        int l= 0,r =nums.size();
        
        while(l<r)
        {   int mid = l+(r-l)/2;
            if(nums[mid]>=target)
            {
                r = mid;
            }
            else{
                l = mid+1;
            }
        }
        return l; 
    }
    int findSec(vector<int>& nums, int target) {
        int l= 0,r =nums.size();
      
        while(l<r)
        {   int mid = l+(r-l)/2;
            if(nums[mid]>target)
            {
                r = mid;
            }
            else{
                l = mid+1;
            }
        }
        return l; 
    }
    vector<int> searchRange(vector<int>& nums, int target)
    {
        int l= findFirst(nums,target);
        int r= findSec(nums,target);
        if(l == nums.size() || nums[l]!=target) return {-1,-1};

        return{l,r-1};
        
        
    }
};
```
1. if(nums[mid]>=target) 第一个大于等于的 ：if(nums[mid]>target) ：第一个大于的

