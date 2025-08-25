# 1. 两数之和
给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。

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

2.用map的key做sum，作value : unordered_map<long long, int> cnt; // 前缀和 -> 出现次数

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
记住：

1.看到滑动窗口最大值，用双端队列deque，保存下标。并保持对应的值从队头到队尾递减。

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

2. 别忘最后需要将curr加入。 

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


# 94
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
