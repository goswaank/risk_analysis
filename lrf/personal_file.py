def getLongestPalindrom(s):
    s_len = len(s)
    new_str = ''

    ## Adding # in between the words
    for i,c in enumerate(s):
        if i!=(s_len-1):
            new_str = new_str + c + '#'
        else:
            new_str = new_str + c

    palindrome_size = []
    for i, c in enumerate(new_str):
        left = new_str[0:i]
        right = new_str[i + 1:len(new_str)]
        left_len = len(left)
        right_len = len(right)
        max_len = left_len if left_len < right_len else right_len
        palindrome_flag = False
        cnt = 0

        while max_len - cnt > 0 and palindrome_flag != True:

            left_tmp = left[left_len - max_len + cnt:left_len]
            right_tmp = right[0:max_len - cnt]
            rev_right = right_tmp[::-1]
            if left_tmp == rev_right:
                palindrome_flag = True
                palindrome_size.append((left_tmp + new_str[i] + right_tmp).replace('#', ''))
            else:
                cnt = cnt + 1


    longestPalindrome = max(palindrome_size, key=len)
    return longestPalindrome


def main():
    s = 'abczzdfgfdz'
    result = getLongestPalindrom(s)

    print(result)

if __name__=='__main__':
    main()