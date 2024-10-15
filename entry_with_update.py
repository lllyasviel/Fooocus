import os
import sys


root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)


try:
    import pygit2
    pygit2.option(pygit2.GIT_OPT_SET_OWNER_VALIDATION, 0)

    repo = pygit2.Repository(os.path.abspath(os.path.dirname(__file__)))

    branch_name = repo.head.shorthand

    remote_name = 'origin'
    remote = repo.remotes[remote_name]

    diff = False

    # Check if the remote branch is ahead of the local branch
    remote_to_check = 'main'
    local_to_check = 'main'
    remote_ref = f'refs/remotes/{remote_name}/{remote_to_check}'
    local_ref = f'refs/heads/{local_to_check}'
    remote_commit = repo.revparse_single(remote_ref)
    local_commit = repo.revparse_single(local_ref)
    if remote_commit.id != local_commit.id:
        diff = True

    user_wants_to_update = False
    if diff:
        if input("The remote branch is ahead of the local branch. Do you want to update? (y/n): ") != 'y':
            user_wants_to_update = True
            remote.fetch()

    if user_wants_to_update:
        local_branch_ref = f'refs/heads/{branch_name}'
        local_branch = repo.lookup_reference(local_branch_ref)

        remote_reference = f'refs/remotes/{remote_name}/{branch_name}'
        remote_commit = repo.revparse_single(remote_reference)

        merge_result, _ = repo.merge_analysis(remote_commit.id)

        if merge_result & pygit2.GIT_MERGE_ANALYSIS_UP_TO_DATE:
            print("Already up-to-date")
        elif merge_result & pygit2.GIT_MERGE_ANALYSIS_FASTFORWARD:
            local_branch.set_target(remote_commit.id)
            repo.head.set_target(remote_commit.id)
            repo.checkout_tree(repo.get(remote_commit.id))
            repo.reset(local_branch.target, pygit2.GIT_RESET_HARD)
            print("Fast-forward merge")
        elif merge_result & pygit2.GIT_MERGE_ANALYSIS_NORMAL:
            print("Update failed - Did you modify any file?")

        print('Update succeeded.')
except Exception as e:
    print('Update failed.')
    print(str(e))

print("Launching the application...")
from launch import *
