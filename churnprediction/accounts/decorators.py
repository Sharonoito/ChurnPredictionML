from django.core.exceptions import PermissionDenied

def role_required(allowed_roles):
    def decorator(view_func):
        def _wrapped_view(request, *args, **kwargs):
            if request.user.role in allowed_roles:
                return view_func(request, *args, **kwargs)
            else:
                raise PermissionDenied  # Deny access if role is not allowed
        return _wrapped_view
    return decorator

