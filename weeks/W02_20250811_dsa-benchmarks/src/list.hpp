#pragma once
#include <cstddef>
#include <utility>
#include <cassert>

template <typename T>
class SinglyList
{
    struct Node
    {
        T v;
        Node *next;
    };

public:
    SinglyList() : head_(nullptr), n_(0) {}
    ~SinglyList() { clear(); }

    void push_front(const T &v)
    {
        head_ = new Node{v, head_};
        ++n_;
    }
    void push_front(T &&v)
    {
        head_ = new Node{std::move(v), head_};
        ++n_;
    }

    // Removes first node, returns true if removed.
    bool pop_front()
    {
        if (!head_)
            return false;
        Node *t = head_;
        head_ = head_->next;
        delete t;
        --n_;
        return true;
    }

    template <typename F>
    void for_each(F &&f) const
    {
        for (Node *p = head_; p; p = p->next)
            f(p->v);
    }

    std::size_t size() const { return n_; }
    void clear()
    {
        while (head_)
        {
            Node *t = head_;
            head_ = head_->next;
            delete t;
        }
        n_ = 0;
    }

private:
    Node *head_;
    std::size_t n_;
};
