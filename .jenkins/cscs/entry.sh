#!/bin/bash -l

# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Make undefined variables errors, print each command
set -eux

project_name=hpx-kokkos

# Clean up old artifacts
rm -f ./jenkins-${project_name}* ./*-Testing

export configuration_name_with_options="${configuration_name}-${build_type,,}-hpx-${hpx_version}-kokkos-${kokkos_version}-future-type-${future_type}"

source .jenkins/cscs/slurm-constraint-${configuration_name}.sh

if [[ -z "${ghprbPullId:-}" ]]; then
    # Set name of branch if not building a pull request
    export git_local_branch=$(echo ${GIT_BRANCH} | cut -f2 -d'/')
    job_name="jenkins-${project_name}-${git_local_branch}-${configuration_name_with_options}"
else
    job_name="jenkins-${project_name}-${ghprbPullId}-${configuration_name_with_options}"

    # Cancel currently running builds on the same branch, but only for pull
    # requests
    scancel --account="djenkssl" --jobname="${job_name}"
fi

# Start the actual build
set +e
sbatch \
    --job-name="${job_name}" \
    --nodes="1" \
    --constraint="${configuration_slurm_constraint}" \
    --partition="cscsci" \
    --account="djenkssl" \
    --time="00:30:00" \
    --output="jenkins-${project_name}-${configuration_name_with_options}.out" \
    --error="jenkins-${project_name}-${configuration_name_with_options}.err" \
    --wait .jenkins/cscs/batch.sh

# Print slurm logs
echo "= stdout =================================================="
cat jenkins-${project_name}-${configuration_name_with_options}.out

echo "= stderr =================================================="
cat jenkins-${project_name}-${configuration_name_with_options}.err

# Get build status
status_file="jenkins-${project_name}-${configuration_name_with_options}-ctest-status.txt"
if [[ -f "${status_file}" && "$(cat ${status_file})" -eq "0" ]]; then
    github_commit_status="success"
else
    github_commit_status="failure"
fi

if [[ -n "${ghprbPullId:-}" ]]; then
    # Extract just the organization and repo names "org/repo" from the full URL
    github_commit_repo="$(echo $ghprbPullLink | sed -n 's/https:\/\/github.com\/\(.*\)\/pull\/[0-9]*/\1/p')"

    # Get the CDash dashboard build id
    cdash_build_id="$(cat jenkins-${project_name}-${configuration_name_with_options}-cdash-build-id.txt)"

    # Extract actual token from GITHUB_TOKEN (in the form "username:token")
    github_token=$(echo ${GITHUB_TOKEN} | cut -f2 -d':')

    # Set GitHub status with CDash url
    .jenkins/common/set_github_status.sh \
        "${github_token}" \
        "${github_commit_repo}" \
        "${ghprbActualCommit}" \
        "${github_commit_status}" \
        "${configuration_name_with_options}" \
        "${cdash_build_id}" \
        "jenkins/cscs"
fi

set -e
exit $(cat ${status_file})
