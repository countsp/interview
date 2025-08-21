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
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> ans;
        
        sort(nums.begin(),nums.end());
        for(int l =0;l<nums.size()-2;++l)
        {
            int mid = l+1; int r = nums.size()-1; int sum = -nums[l];
            while(mid<r)
            {   while(mid)
                if(nums[mid]+nums[r] == sum)
                    {ans.push_back({nums[l],nums[mid],nums[r]});mid++;r--;while(mid<r && nums[mid]==nums[mid-1]) {mid++;} while(mid<r && nums[r]==nums[r+1]) {r--;}}
                else if(nums[mid]+nums[r] > sum){r--;}
                else if(nums[mid]+nums[r] < sum){mid++;}
                
                
            }
        }
    return ans;
}
};
```

---

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
