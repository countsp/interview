### quickSort

```
void quickSort(vector<int>& a, int lo, int hi) {
    if (lo >= hi) return;
    int pivotVal = a[hi];  // 枢轴值
    int i = lo;
    for (int j = lo; j < hi; ++j) {
        if (a[j] < pivotVal) {
            swap(a[i], a[j]);
            ++i;
        }
    }
    swap(a[i], a[hi]);
    quickSort(a, lo, i - 1);
    quickSort(a, i + 1, hi);
}

void quickSort(vector<int>& a) {
    if (!a.empty())
        quickSort(a, 0, a.size() - 1);
}
```

###  heapSort

```
#include <vector>
#include <algorithm>
using namespace std;

// 对以 a 为底数组、长度为 n、以 i 为根的子树执行下沉（heapify）
// 保证以 i 为根的子树满足大顶堆性质
void heapify(vector<int>& a, int n, int i) {
    int largest = i;        // 假设根最大
    int left    = 2*i + 1;  // 左子节点索引
    int right   = 2*i + 2;  // 右子节点索引

    // 如果左子节点比根大
    if (left < n && a[left] > a[largest])
        largest = left;
    // 如果右子节点比当前最大还大
    if (right < n && a[right] > a[largest])
        largest = right;
    // 如果最大不是根，则交换并递归下沉
    if (largest != i) {
        swap(a[i], a[largest]);
        heapify(a, n, largest);
    }
}

// 堆排序主函数
void heapSort(vector<int>& a) {
    int n = a.size();
    if (n < 2) return;

    // 1. 建立大顶堆：从最后一个非叶子节点开始向前下沉
    for (int i = n/2 - 1; i >= 0; --i) {
        heapify(a, n, i);
    }

    // 2. 排序：不断把堆顶（最大）移到末尾，然后下沉恢复堆
    for (int i = n - 1; i > 0; --i) {
        // 交换堆顶和当前堆的末尾
        swap(a[0], a[i]);
        // 对剩余 [0..i-1] 区间重新下沉调整
        heapify(a, i, 0);
    }
}
```



### mergeSort

void mergeSort(vector<int>& a,int lo,int hi)

{   

​    if(lo>=hi)return;

​    int mid = lo+(hi-lo)/2;

​    mergeSort(a,lo,mid);

​    mergeSort(a,mid+1,hi);



​    vector<int> ans;

​    ans.reserve(hi-lo+1);

​    int m =lo,n=mid+1;

​    while(m<=mid && n<=hi)

​    {

​        if(a[m]<=a[n]){ans.push_back(a[m++]);}

​        else {ans.push_back(a[n++]);}

​    }

​    while(m<=mid){ans.push_back(a[m++]);};

​    while(n<=hi){ans.push_back(a[n++]);};

​    

​    for (int k = 0; k < ans.size(); ++k) {

​    a[lo + k] = ans[k];

​    }

}

void mergeSort(vector<int>& a) {

​    if (a.size() > 1) 

​        mergeSort(a, 0, int(a.size()) - 1);

}

 ## quickSort在分块时，不包含mid。但是mergeSort包含。